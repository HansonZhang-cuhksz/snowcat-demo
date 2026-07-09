# Implementation Plan — `decode_area_latency.py` (full decode layer: pre-attn norm + MLA + residual + FFN)

Goal: a new source file `decode_area_latency.py` that expands `ffn_area_latency.py` from the
decode **FFN** stage to a **full GLM-5.2 decode transformer layer**:

```
pre-attention RMSNorm  ->  MLA (multi-head latent attention)  ->  residual add
      ->  (existing) pre-FFN RMSNorm  ->  MoE FFN (router, up_gate, down, SwiGLU, expert combine)
      ->  (existing) post-FFN residual add
```

`ffn_area_latency.py` is left untouched. The new file is a superset copy with the attention stages
added. Same two estimation tools as the rest of the repo: Snowcat/Orojenesis Pareto traffic frontier
+ latency-aware roofline with per-kernel `num_stages` (see `latency_pipeline_model.md`).

## Config changes vs. ffn_area_latency.py

- `BATCH_TOKENS = 1024` (was 4096)
- `TOKENS_PER_EXPERT = 32` (was 128) — 1024 tokens * top_k 8 / 256 experts = 32/expert. `ROUTER_TOP_K`
  stays 8 (= EXPERTS*TOKENS_PER_EXPERT/BATCH_TOKENS = 256*32/1024).
- `SEQ_LEN = 100_000` (new) — KV-cache context length per sequence.
- MLA constants (GLM-5.2, from HF config.json — see memory `glm-5-2-config`):
  `N_HEADS=64, KV_LORA_RANK=512, Q_LORA_RANK=2048, QK_NOPE_HEAD_DIM=192, QK_ROPE_HEAD_DIM=64,
  V_HEAD_DIM=256`. Derived: `QK_HEAD_DIM = 192+64 = 256`, `KV_LATENT = 512+64 = 576` (cached/token).
- `KV_SPARSITY_FACTOR = 1.0` — dense MLA (=1.0 as requested). GLM-5.2 really uses DSA; set <1 to model
  the sparse-attention fraction of KV positions actually scored. Default 1.0.
- `ATTENTION_SOFTMAX_FLOPS_PER_ELEMENT = 5.0` — CUDA-core cost per (head,pos) softmax element (placeholder).
- `ATTN_KV_BLOCK = 16` — flash-decode KV position block per pipeline stage (for the latency `num_stages`
  reporting on the streaming attention core).

Everything else (chip area constants, TENSOR_FLOPS, CUDA clock, HBM latency/bw, area grid,
register-accumulator mapspace, random-expert path) is copied unchanged.

## MLA decode model (matrix-absorbed / flash-decode)

Decode caches only the latent `c_KV` (KV_LORA_RANK) + decoupled RoPE key (QK_ROPE_HEAD_DIM) = 576
elem/token; K and V are never materialized per head. Batch B=1024 sequences, each with an independent
S=100k latent cache (per-sequence, BF16). Two parts:

### A. MLA weight GEMMs — routed through the snowcat frontier + latency roofline exactly like FFN GEMMs

Added as explicit `GemmTaskGroup`s (M = batch, so batch lives in M; `count` as noted).
`GemmTask(name, m, n, k)`:

| name | M | N | K | count | note |
|---|---|---|---|---|---|
| `mla_q_a`  | B | Q_LORA_RANK=2048 | HIDDEN=6144 | 1 | q down-proj |
| `mla_q_b`  | B | N_HEADS*QK_HEAD_DIM=16384 | Q_LORA_RANK=2048 | 1 | q up-proj |
| `mla_kv_a` | B | KV_LORA_RANK+QK_ROPE=576 | HIDDEN=6144 | 1 | kv down-proj (+k_rope) |
| `mla_wuk_absorb` | B | KV_LORA_RANK=512 | QK_NOPE_HEAD_DIM=192 | N_HEADS=64 | W_UK absorbed into q_nope |
| `mla_wuv_absorb` | B | V_HEAD_DIM=256 | KV_LORA_RANK=512 | N_HEADS=64 | W_UV absorbing attn latent |
| `mla_o`    | B | HIDDEN=6144 | N_HEADS*V_HEAD_DIM=16384 | 1 | output proj |

These are weight-static GEMMs; they get the same per-point best-`C` (`num_stages`) treatment via the
existing `build_traffic_frontier` / `gemm_time_from_frontier` / `select_mapping_from_frontier`. They
are small (~hundreds of MiB weight traffic) vs. the attention core, but included for completeness.

### B. MLA attention core — custom fused flash-decode task (KV-cache streaming), latency-aware

NOT routed through `enumerate_mappings` (a) to keep flash fusion (scores never spilled to HBM) and
(b) to avoid enumerating over N=seq_len. `count = B` (each sequence streams its own cache). Per
sequence, over S positions (scaled by KV_SPARSITY_FACTOR):

- scores (tensor): `q_combined[H,576] . KVcombined[576,S]`  -> ops = 2*H*S*KV_LATENT
- output/AV (tensor): `attn[H,S] . cKV[S,512]`            -> ops = 2*H*S*KV_LORA_RANK
- softmax (cuda): ~ATTENTION_SOFTMAX_FLOPS_PER_ELEMENT * H*S
- traffic (fused, KV read once): `S * KV_LATENT * 2` (c_KV+k_rope), + small q read + attn-latent write.

Time (mirrors the fused-stage roofline): `count * max(tensor_ops/tensor_roof, cuda_ops/cuda_roof,
traffic/BW_eff)`. Memory term uses the latency-aware `BW_eff`: streaming fills SMEM with in-flight KV
so `inflight ~= S_total`, but reported via the note's formula with a nominal stage buffer
`W_stage = ATTN_KV_BLOCK*KV_LATENT*2`: `C_best=min(floor(S_total/W_stage), ceil(bw*latency/W_stage))`,
`BW_eff=min(bw, C_best*W_stage/latency)`. At the FFN-optimal area (~2.26 MiB SMEM) BW saturates
(`bw*latency ~= 706 KiB < S_total`) so `BW_eff -> bw`.

Sanity: KV traffic = B*S*576*2 = 1024*100000*576*2 = 118 GB -> /2.04 TB/s ~= 58 ms, which dominates
the ~10 ms FFN. Attention core OI = (2H(576+512))/(576*2) ~= 121 FLOP/byte (memory-bound at the FFN
area point). Confirms long-context batched decode is KV-bandwidth-bound — the expected result.

### C. New vector stages (reuse existing task types)

- `pre_attention_rmsnorm`: `ReductionTask(rows=B, columns=HIDDEN, bpe_in=2, bpe_out=4)` — identical
  shape to the existing pre-FFN `RMSNORM_SQUARE_REDUCTION_TASK`, time via `reduction_time`.
- `post_attention_residual_add`: `VectorTask(elements=B*HIDDEN, count=1, flops=1, traffic=3*2)` —
  identical to the existing post-FFN `RESIDUAL_ADD_TASK`, time via `vector_time`.

## Plumbing changes (mostly extend existing functions)

1. Prepend MLA GEMM groups to `task_groups` in `main()` (both deterministic and random-expert paths;
   attention is independent of expert routing). They flow through `build_frontiers` and the existing
   per-frontier time/traffic loop unchanged. `group_gemm_tasks` is only used for the FFN tasks (its
   EXPERTS-multiply logic must not touch MLA groups — hence MLA groups are built explicitly).
2. `attention_core_time(s_total, tensor_roof, cuda_roof) -> (time[], traffic[])` and
   `attention_core_mapping(...)` for the best-node report (num_stages + BW_eff + OI).
3. Add `pre_attention_rmsnorm`, attention core, `post_attention_residual_add` into: `total_time`
   sum, `modeled_operations`, `total_hbm_traffic_bytes`, `write_csv` columns, and the printed report
   (new "=== Attention Stages ===" section + the two new vector lines).
4. `output_paths()` -> `./result/decode_area_latency{suffix}_*.{csv,png}`.
5. Keep `ffn_area_latency.py` unchanged.

## Out of scope (documented, not modeled)

- Shared expert (n_shared_experts=1), DSA sparse-attention selection/indexer (KV_SPARSITY_FACTOR knob
  only), internal q/kv latent RMSNorms (tiny), RoPE application cost (tiny), MTP.

## Verification

1. `conda run -n base python decode_area_latency.py` exits 0; CSV + PNGs written. (NOTE: the
   `area` conda env referenced in older notes no longer exists; `base` has numpy 2.4.4 + matplotlib.)
2. Attention core dominates total time and total HBM (long-context KV-bound). CONFIRMED.
3. FFN M-scaled GEMMs shrink vs. the 4096-batch report (batch 4096->1024). CONFIRMED.
4. Each MLA GEMM prints a mapping line with `num_stages`; attention core prints num_stages + BW_eff.
   CONFIRMED.
5. Random-expert path (`USE_RANDOM_EXPERT_DISTRIBUTION=True`): NOT supported at batch 1024 — the
   binomial tail includes experts with <16 tokens, for which no tensor-core tile (BM>=16) exists, so
   `build_traffic_frontier` raises. This is a pre-existing fragility of the random path (fine at
   batch 4096 where the tail stays >=16), exposed by the smaller batch. The specified workload is
   even distribution (default `False`), which works. Out of scope to fix (would need M-padding to 16
   or dropping negligible-probability tiny-M experts).

## Results (even distribution, default; register-accumulator mapspace, HBM latency 500 cyc)

Best area node: rc=0.018, rt=0.976, SMEM frac 0.006 -> **1.128 MiB**, 490 CUDA cores, 886 tensor
cores. **Total 68.47 ms**, total HBM **133.1 GB**, effective **222.8 TFLOP/s**.

| Stage | time | HBM | note |
|---|---:|---:|---|
| mla_attention (core) | 57.90 ms | 118.1 GB | memory-bound; OI 121, num_stages 40, BW_eff=bw. Dominates. |
| up_gate_x256 + down_x256 | 6.40 + 3.22 ms | 12.4 + 6.3 GB | FFN GEMMs, memory-bound |
| mla_o / q_b / q_a / wuv / wuk / kv_a | 0.45/0.15/0.06/0.06/0.05/0.02 ms | ~1.4 GB total | MLA projections |
| router | 0.008 ms | 15.5 MiB | |
| pre/post-attn norm+residual, FFN norm/act/combine/residual | <0.06 ms each | small | |

Hand-checked: attention mem 118.1 GB / 2.04 TB/s = 57.9 ms; tensor 14.26 TFLOP / (886*512 GFLOP/s
=453.6 TFLOP/s) = 31.4 ms; softmax 32.8 GFLOP / (490*5.64 GFLOP/s) = 11.9 ms -> max = 57.9 ms.
Best SMEM dropped from the FFN-only 2.257 MiB to 1.128 MiB (attention saturates BW at ~706 KiB =
bw*latency), freeing area for tensor cores (886 vs 880). Whole layer is essentially HBM-bandwidth
bound (total HBM 133 GB / 2.04 TB/s ~= 65 ms of the 68 ms).

## Parametrization + 1M-token batch sweep (follow-up)

Env: the `profiling` conda env is the intended one (dev machine). It lacked numpy/matplotlib
here; installed `numpy 2.5.1` + `matplotlib 3.11.0` into it (`pip`). Run with
`conda run -n profiling python ...`. (`base` also works.)

Refactor of `decode_area_latency.py` (behavior-preserving — batch 1024 / seq 100k still gives the
exact 68.468 ms / rc0.018 / rt0.976 / 1.128 MiB result):
- `ROUTER_TOP_K` is now the fixed input (8); `TOKENS_PER_EXPERT = batch*top_k/experts` is derived.
- `configure(batch_tokens, seq_len)` (re)binds all workload-derived globals; called at import with
  the module defaults. Defaults are now `DEFAULT_BATCH_TOKENS=2048`, `DEFAULT_SEQ_LEN=1_048_576`
  (GLM-5.2 max context; batch chosen since the optimal split is batch-invariant). Bare
  `python decode_area_latency.py` -> best split rc0.018/rt0.975/r_smem0.007 (490 CUDA, 885 tensor,
  1.316 MiB), total 1224.5 ms, 2382 GiB HBM, attn 99.0%. (`USE_REGISTER_ACCUMULATOR_MAPPINGS=True`;
  note the user set `CPU_WORKERS=8`.)
- `evaluate_layer()` extracted (pure compute -> results dict incl. `best_index`); `main(write_outputs=True)`
  reports + optionally writes. CLI: `--batch-tokens`, `--seq-len`, `--no-write`.
- New `decode_area_sweep.py`: sweeps batch sizes in-process at a fixed seq_len (default 1M, batches
  512/1024/2048/4096). `conda run -n profiling python decode_area_sweep.py`.

### Best area distribution at seq_len = 1,000,000 ("absolute toughest")

| batch | tok/exp | rc | rt | r_smem | SMEM | CUDA | Tensor | total | attn% | HBM | TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 16 | 0.018 | 0.976 | 0.006 | 1.128 MiB | 490 | 886 | 0.299 s | 96.6% | 568 GiB | 240.5 |
| 1024 | 32 | 0.018 | 0.976 | 0.006 | 1.128 MiB | 490 | 886 | 0.589 s | 98.2% | 1119 GiB | 244.3 |
| 2048 | 64 | 0.018 | 0.975 | 0.007 | 1.316 MiB | 490 | 885 | 1.168 s | 99.0% | 2219 GiB | 246.3 |
| 4096 | 128 | 0.018 | 0.975 | 0.007 | 1.316 MiB | 490 | 885 | 2.327 s | 99.4% | 4421 GiB | 247.3 |

**The optimal split is batch-invariant: ~1.8% CUDA / ~97.5% tensor / ~0.6-0.7% SMEM** (490 CUDA,
~885 tensor, ~1.1-1.3 MiB SMEM). Reason: the layer is dominated by memory-bound MLA KV-cache
streaming, whose memory time is area-independent once BW is saturated. So the split only needs
(a) enough tensor cores that attention's QK/AV time (attn_tensor) stays below attn_mem, (b) enough
SMEM (>= bw*latency ~= 706 KiB) to saturate HBM BW, (c) enough CUDA cores for softmax (attn_softmax).
Per-stage (batch 4096): attn_mem 2313 ms >> attn_tensor 1259 ms > attn_softmax 474 ms > up_gate
6.6 ms (FFN weight traffic, ~batch-invariant) > down 3.4 ms. Attention (KV) scales linearly with
batch while FFN weight streaming is fixed, so attention's dominance grows with batch (96.6 -> 99.4%).
Total time & HBM scale linearly with batch. "1M" taken as 1,000,000 (GLM-5.2 max context 1,048,576).

## Sensitivity study + report (follow-up)

Default now batch 2048 / seq 1,048,576. Sensitivity of total decode-layer time (area re-optimized per
point) at that default:
- **Bandwidth (x0.5..x2)**: time ∝ 1/bw exactly (time×bw ≈ 2497 const 1.02→3.06 TB/s; throughput linear
  in bw: 123/185/246/369 TFLOP/s). Deviates only at 4.08 TB/s (660 vs ~612 ms) — SMEM budget caps
  attn num_stages at 74 so BW_eff=3.846<4.08; past ~3 TB/s the design must spend more area on SMEM.
- **Latency (x0.5..x8, 250→4000 cyc)**: flat (1224.49→1224.53 ms). Optimizer deepens attn pipeline
  (num_stages 20→314) + buys SMEM (0.75→5.83 MiB), BW_eff pinned at 2.04.
- **Tensor throughput (x0.5..x2)**: flat except +7% at 0.5x (256 GF/s → 1312 ms, GEMMs go compute-bound).
- **CUDA throughput (x0.5..x2)**: no effect (±0.04 ms); softmax/vector are negligible.

Confirms HBM-bandwidth bound. Deliverable: `decode_area_report.md` (copied from decode_ffn_area_report.md,
same style) with Assumptions/Workloads/Area+Stage Results/Graphs + the 4 sensitivity tables
(Bandwidth/Latency/Tensor/CUDA). Plots regenerated to `img/decode_area_latency_{total,attention}_time.png`.
Experiment driver: scratch `tmp/experiments.py` (reassigns module globals bw/HBM_LATENCY_CYCLES/
TENSOR_FLOPS/ACTIVATION_FLOPS_PER_CUDA_CORE then re-runs evaluate_layer).

## STATUS: DONE (even-distribution path; parametrized; 1M sweep; sensitivity study + decode_area_report.md)
