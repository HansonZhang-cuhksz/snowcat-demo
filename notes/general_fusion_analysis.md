# General Fusion Area-Distribution Analysis — Master Plan

Goal: for each of six kernel fusions, build a custom **latency-aware roofline +
Snowcat/Orojenesis** model of the *fused* kernel, run the die-area distribution
analysis over the **full GLM-5.2 decode layer** with **only that one kernel fused**
(everything else unfused), and compare the optimal area split / time / traffic to the
fully-unfused baseline.

Confirmed scope decisions (user, 2026-07-16):
1. **Baseline = full decode layer** (`decode_area_latency.py`): pre-attn RMSNorm → MLA
   flash-decode → residual → pre-FFN RMSNorm → MoE FFN (router, up_gate+SwiGLU, down,
   combine) → residual. Only the target kernel is fused per analysis; all comparable on
   one layer. (The MLA KV-cache read is ~99% of layer time, so FFN-region fusions barely
   move the *layer* optimum — that null result is itself a reported finding; the per-stage
   effect is reported alongside.)
2. **"FlashAttention" = decode MLA flash-decode** (matrix-absorbed attention core +
   its output projection `mla_o`), not prefill attention.
3. **Fusion traffic model = CODA on-chip intermediate** (same register-accumulator style
   as `coda_fused_traffic.py`): the intermediate tensor between the fused ops is never
   written to / read from HBM; compute terms sum; the one-stage working set `W`
   (`buffer_bytes`) grows to hold the extra epilogue/prologue aux state. Genuine external
   inputs (residual/skip tensor, RMSNorm γ weights) are still read from HBM.

## The six fusions (tasks #5–#10)

| # | Fusion | Fused kernel replaces | Baseline stages removed |
|---|--------|-----------------------|-------------------------|
| 1 | FlashAttention + residual | `mla_o` GEMM + post-attn residual add | `mla_o`, `post_attention_residual_add` |
| 2 | FlashAttention + residual + RMSNorm | `mla_o` + residual + pre-FFN RMSNorm | `mla_o`, `post_attention_residual_add`, `rmsnorm_square_reduction` |
| 3 | RMSNorm + up_gate | pre-FFN RMSNorm + up_gate GEMM (prologue norm) | `rmsnorm_square_reduction`, `up_gate` |
| 4 | up_gate + activation | up_gate GEMM + SwiGLU epilogue | `up_gate`, `activation` |
| 5 | activation + down | SwiGLU + down GEMM (prologue activation) | `activation`, `down` |
| 6 | up_gate + activation + down | up_gate + SwiGLU + down (single kernel) | `up_gate`, `activation`, `down` |

Notes on direction:
- RMSNorm needs the full row to form the 1/rms scale. Fusing it into the *following*
  GEMM's prologue (fusion 3) or an epilogue of the *preceding* GEMM (fusion 2, partial
  RMS) follows the CODA reparameterization (`gemm_residual_partial_rms_weight`,
  `gemm_rms_scale`).
- Fusion 6 ≈ the existing `ffn_fused_area_latency.py` up_gate→down register-accumulator
  fusion, but re-hosted inside the full decode layer for a like-for-like comparison.

## Shared method / architecture (repo cleanliness)

```
fusion/
  common.py                      # reusable latency engine (NOT fusion-specific)
  <fusion_name>/
    plan.md                      # per-fusion plan (written before implementing)
    model.py                     # the custom fused-kernel model (traffic + compute)
    analysis.py                  # baseline vs fused over the full decode layer + report/plots
    report.md                    # comparison writeup
    log.md                       # step log
    result/                      # images + csv for this fusion
```

- **Reuse, don't duplicate.** The unfused baseline and every unfused stage come straight
  from `decode_area_latency.evaluate_layer()` (imported). A fused analysis computes only
  the one fused kernel and swaps it into the layer total:
  `total_fused = total_baseline − Σ(removed stage times) + fused_kernel_time`,
  then re-argmins for the fused optimum. No re-summing of the unaffected stages.
- **`fusion/common.py`** holds the fusion-agnostic latency machinery:
  - `FusedFrontier` dataclass (`buffer_bytes` W, `traffic_bytes` T, `bm/bn/bk`,
    `loop_orders`, `tensor_operations`, `cuda_operations`, `count`).
  - `build_fused_frontier(points, ...)` — collapse CODA `FusedTraffic` points to the
    Pareto frontier keyed on `W = buffer_bytes` (running-min traffic), matching the
    non-fused frontier shape used across the repo.
  - `fused_stage_time(frontier, s_total, tensor_roof, cuda_roof)` — the per-Pareto-point
    best-`C` (num_stages) latency evaluation (`C_best = min(⌊S/W⌋, ⌈bw·lat/W⌉)`,
    `BW_eff = min(bw, C·W/lat)`, `time = count·max(tensor_ops/tensor_roof,
    cuda_ops/cuda_roof, T/BW_eff)`), vectorized over the area grid. Returns
    `(time, traffic)`.
  - `select_mapping` / `format_mapping` — winning tiling + `num_stages` at the best node.
  - Imports chip constants, `bw`, `HBM_LATENCY_CYCLES`, `CUDA_CLOCK_HZ` from
    `decode_area_latency` so every fusion shares one source of truth.
- **`<fusion>/model.py`** holds the *custom* part: the fused kernel's dims (M,N,K),
  its `tensor_operations` / `cuda_operations`, its CODA traffic-point generator (which
  intermediate is eliminated, which aux buffers/HBM appear), and the list of baseline
  stage-time arrays it removes.
- Runner: `conda run -n fusion python -m fusion.<name>.analysis` from the repo root
  (keeps `snowcat_demo`, `decode_area_latency`, `coda_fused_traffic` importable). Each
  analysis writes only into its own `fusion/<name>/result/`.

## Reference model (unchanged from notes/latency_pipeline_model.md)

```
N = W = buffer_bytes (one-stage working set, incl. output tile + fused aux)
latency = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
C_best  = min(⌊S_total/W⌋, ⌈bw·latency/W⌉)          # smallest optimal num_stages
BW_eff  = min(bw, C_best·W/latency)
time    = count · max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T/BW_eff)
```

Per-Pareto-point best-`C` is exactly the 1-D integer search over `C` (proof in
`notes/latency_pipeline_model.md`). Report `num_stages=C_best (max_feasible=⌊S/W⌋)`.

## Verification per fusion

1. Fused analysis runs clean under `conda run -n fusion`.
2. Fused kernel FLOPs = Σ(constituent kernel FLOPs) (fusion changes traffic/boundaries,
   not compute) — assert `modeled_operations` unchanged vs baseline.
3. Traffic-saved = Σ(eliminated intermediate write+reread), hand-checked against the
   CODA aux accounting.
4. Report both the layer-level optimum shift (often ~null for FFN fusions) and the
   per-stage improvement of the fused kernel.

## STATUS: DONE — all six fusions implemented, run, and cross-checked

Full results + figure: `fusion/SUMMARY.md`. Per-fusion detail: `fusion/<name>/report.md`.
Baseline optimum (reproduced exactly by every analysis): 490 CUDA / 885 tensor /
1.316 MiB SMEM, 1224.492 ms, 2326.161 GiB HBM.

| # | Fusion | HBM saved | Optimal CUDA/Tensor | FLOPs |
|---|--------|---:|:--:|:--:|
| 1 | FA + residual | 48.000 MiB | 490 / 885 | conserved |
| 2 | FA + residual + RMSNorm | 71.867 MiB | 490 / 885 | conserved |
| 3 | RMSNorm + up_gate | 24.008 MiB | 490 / 885 | +0.02% redundant |
| 4 | up_gate + activation | 256.000 MiB | 381 / 889 | conserved |
| 5 | activation + down | 128.000 MiB | 381 / 889 | conserved |
| 6 | up_gate + activation + down | 384.000 MiB | 354 / 890 | conserved |

Headline findings: (a) all fusions are strictly-positive HBM wins but ≤0.017% time wins
(layer is KV-cache bound); (b) FFN fusions that remove the CUDA-core SwiGLU (4/5/6) shift
the die partition toward tensor cores (490→381→354 CUDA); (c) epilogue > prologue (F4
256 MiB > F5 128 MiB for the same SwiGLU); (d) F2 > F3 for removing the same RMSNorm; (e)
F6 (full FFN) is strongest but SMEM-gated (needs ~1 MiB to hold the m0=64 row-block, else
weight re-reads dominate — why the repo's ffn_fused leaves down un-fused).

Engine (`fusion/common.py`) was adversarially verified by two workflows (engine+F1, and
F3–F6 custom models). Env fix: `StrEnum` compat shim in `snowcat_demo/model/decision.py`
for the 3.10 `fusion` env (no-op on 3.11+).

### Notes / caveats for a later pass
- F3 models the inline RMS with `aux_hbm=0` (γ folded into W, scale computed from A reads);
  intentional ×top_k redundant recompute (negligible).
- F6 enumerates only the row-block `m0` (weights read once per block, full accumulators
  resident); finer per-GEMM sub-tilings would add intermediate frontier points but the
  `m0` knob captures the dominant weight-reread-vs-SMEM tradeoff.
- All analyses use the even-expert-split single-group up_gate/down (the reported config);
  the random/uneven-expert path is not wired into the fusion swap (documented extension).

## STATUS: PREFILL — all six fusions also run (2026-07-16)

Full results + figure: `fusion/SUMMARY_prefill.md`. Per-fusion: `fusion/<name>/report_prefill.md`.
The engine (`fusion/common.py`) and all six `model.py` were refactored to **inject the
baseline module** (`build_frontier(bl)`, `run_fusion(model, baseline, ...)`), so the same
models serve decode (`decode_area_latency`) and prefill (`prefill_area_latency`) — no
duplication. Decode runners are `fusion/<name>/analysis.py`; prefill runners are
`fusion/<name>/analysis_prefill.py` (outputs to `result/prefill_*`).

**HBM convention change (applies to both stages):** HBM is now reported as **min traffic
among time-optimal tilings** (`baseline_gemm_min_traffic` + `layer_total_and_removed_traffic`
+ lexicographic `fused_stage_time`). This is required because prefill is compute-bound —
many tilings tie on time, so the raw frontier traffic (which the baseline module reports as
the *highest*-traffic tied tiling) is ill-defined. For memory-bound decode stages the
convention is a no-op; it only changed the two compute-bound `mla_o` fusions (decode F1/F2),
which now correctly show a small HBM *increase* (the residual epilogue's on-chip tile
starves the big GEMM) instead of the earlier naive "saving". Decode splits/times unchanged.

Prefill baseline optimum: 326 CUDA / 858 tensor / 8.086 MiB, 12992 ms, 434.5 TFLOP/s
(≈ tensor roof, compute-bound). Prefill results:

| # | Fusion | Time saved | Layer HBM Δ | Split |
|---|--------|---:|---:|:--:|
| 1 | FA + residual | 18.95 ms | +72 GiB | 326/858 |
| 2 | FA + residual + RMSNorm | 25.96 ms | +60 GiB | 326/858 |
| 3 | RMSNorm + up_gate | 7.01 ms | −12 GiB | 326/858 |
| 4 | up_gate + activation | 74.75 ms | −128 GiB | 326/858 |
| 5 | activation + down | 74.75 ms | +32 GiB | 326/858 |
| 6 | up_gate + activation + down | 74.75 ms | +384 GiB | 326/858 |

Headline prefill findings: (a) **no fusion moves the die split** (tensor-bound; CUDA count
set by the DSA indexer, not the FFN) — the sharpest decode/prefill contrast; (b) time wins
are just "remove the small vector/CUDA kernel" (≤0.58%); (c) HBM is hidden and often
*increases* (F1/F2 starvation, F5 down-prologue widening, **F6 +384 GiB weight-reread
catastrophe at M=32768** — the quantitative reason full-FFN fusion is decode-only); (d)
epilogue still beats prologue (F4 −128 vs F5 +32 GiB for the same SwiGLU).

## STATUS: HARDWARE SENSITIVITY-TO-FUSION SWEEP (2026-07-16)

Full report + figure: `fusion/SENSITIVITY.md`, `result/sensitivity.png`, `result/sensitivity.json`.
Harness `fusion/sensitivity.py` (sweeps HBM bw/latency, tensor GFLOP/s, CUDA GFLOP/s one at a
time; reuses the hardware-independent frontiers via caching; coarse 0.002 grid → Δsplit
quantised ~55 cores). Validated: the swept unfused optima reproduce the models' own
sensitivity tables (decode tensor=1024→871, =256→900; prefill→846).

Findings: **Die-split sensitivity to fusion is a decode (memory-bound) phenomenon, driven
by CUDA-core throughput** — weak CUDA cores (2.82 GFLOP/s) → the activation-removing fusions
(F4/F5/F6) swing the optimal CUDA count by ~327 cores; strong CUDA (11.28) → ~109. It also
needs moderate–high bandwidth and ≥384 GFLOP/s tensor; it vanishes at ≥4 TB/s bandwidth or
weak tensor. Attention fusions (F1/F2/F3) never move the split. **Prefill split is
fusion-insensitive** (except at 256 GFLOP/s tensor). **Time**: prefill ~30× more
fusion-sensitive than decode (~0.5–0.77% vs ~0.02%), growing at low bandwidth / high tensor;
**HBM latency is irrelevant**; and at 4.08 TB/s decode F6 flips to −0.18% (fusion *hurts*).

## STATUS: DECODE BATCH-SIZE SWEEP (2026-07-16)

Full report + figure: `fusion/BATCH_SWEEP.md`, `result/batch_sweep.png/.json`. Harness
`fusion/batch_sweep.py` (sweeps batch 32..32768 at default GPU spec + 1M context;
re-`configure()`s + rebuilds frontiers per batch; batch<32 excluded = reduced-active-expert
regime). Default batch 2048 reproduces the prior decode result (F6 best, +0.017%, ~164-core
shift, coarse grid).

Findings: (a) **the optimal fusion changes with batch** — F6 (full FFN) is best at
small/mid batch (tokens/expert ≤128), but **F4 (up_gate+SwiGLU) overtakes at batch ≥8192**
(tokens/expert ≥256) because F6's row-block no longer fits SMEM → both weight matrices
re-read ~4× (F6 dHBM goes negative at batch ≥16384; the same SMEM-gated effect that dooms F6
in prefill, here driven by batch); (b) **fusion's time benefit shrinks with batch**
(+0.09% at batch 32 → +0.01% at 32768) as the batch-independent FFN is swamped by the
batch-scaling attention (attention 66%→99.6%); (c) **F5 (act+down) is negative at small
batch** (−0.02%, down-prologue widening); (d) **the die-split shift is batch-robust**
(~110–164 CUDA cores for F4/F5/F6 at every batch; F1/F2/F3 never). Rule: fuse the whole FFN
(F6) at small decode batch, downgrade to F4 as batch grows.
