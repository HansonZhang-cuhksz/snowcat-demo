# General Fusion Area-Distribution Analysis — Summary (all six fusions)

Each fusion was modeled with a custom latency-aware roofline+Snowcat kernel and evaluated
over the **full GLM-5.2 decode layer** (batch 2048, KV context 1,048,576, 256 experts
top-8) with **only that one kernel fused**, then compared to the fully-unfused baseline
(`decode_area_latency.py`). Method + per-fusion detail: `notes/general_fusion_analysis.md`
and each `fusion/<name>/report.md`.

**Unfused baseline optimum:** rc 0.018 / rt 0.975 / SMEM 1.316 MiB → **490 CUDA / 885
tensor cores**, total **1224.492 ms**, **2325.601 GiB** HBM, 246.367 TFLOP/s.

HBM is reported under the **min-traffic-among-time-optimal** convention (among tilings that
tie on time, count the one with least traffic — what a sane scheduler would run). For the
memory-bound FFN fusions this equals the naive traffic; for the two compute-bound `mla_o`
fusions (F1/F2) it reveals a small HBM *increase* (see finding 6). The **prefill** study is
in `SUMMARY_prefill.md`.

## Results

| # | Fusion | Layer HBM Δ | Layer time saved | Optimal CUDA/Tensor | SMEM | FLOPs |
|---|--------|---:|---:|:--:|---:|:--:|
| — | unfused baseline | — | — | 490 / 885 | 1.316 MiB | — |
| 1 | FlashAttention + residual | +336 MiB (increase) | 0.037 ms (0.0030%) | 490 / 885 | 1.316 MiB | conserved |
| 2 | FlashAttention + residual + RMSNorm | +312 MiB (increase) | 0.049 ms (0.0040%) | 490 / 885 | 1.316 MiB | conserved |
| 3 | RMSNorm + up_gate | −24.008 MiB (saved) | 0.012 ms (0.0010%) | 490 / 885 | 1.316 MiB | +0.02% redundant |
| 4 | up_gate + activation | −256.000 MiB (saved) | 0.138 ms (0.0112%) | **381 / 889** | 1.316 MiB | conserved |
| 5 | activation + down | −128.000 MiB (saved) | 0.072 ms (0.0059%) | **381 / 889** | 1.316 MiB | conserved |
| 6 | up_gate + activation + down | **−370.5 MiB (saved)** | 0.204 ms (0.0167%) | **354 / 890** | **1.128 MiB** | conserved |

![summary](result/fusion_summary.png)

## Findings

1. **Every fusion is a negligible *time* win** at this scale. The GLM-5.2 decode layer is
   ~99% MLA-KV-cache-bandwidth bound (2.3 TiB read per step), so the fusions move the layer
   time by ≤ 0.017%. The FFN fusions (F3–F6) also **save** HBM (24–370 MiB); the two
   compute-bound `mla_o` fusions (F1/F2) slightly **increase** it (finding 6). The fusions
   matter for HBM and (for FFN fusions) area balance, **not** for the decode critical path.

2. **Attention-boundary fusions (1–3) do not move the die partition.** They fold small
   vector/reduction kernels into `mla_o`'s epilogue or up_gate's prologue. Their compute is
   trivial and hides under existing tensor roofs, so the optimal 490/885 split is unchanged.

3. **FFN fusions that remove a CUDA-core kernel (4, 5, 6) shift the die partition toward
   tensor cores.** Removing the SwiGLU activation (a CUDA-core kernel) lowers CUDA demand,
   so the optimizer reallocates: 490→381 (F4/F5) and 490→354 (F6) CUDA cores. This is the
   analysis's headline area-distribution result — fusion changes *how the silicon should be
   partitioned*, even when it barely changes total time.

4. **Placement matters — epilogue beats prologue.** The same SwiGLU saves **256 MiB** fused
   into up_gate's epilogue (F4) but only **128 MiB** fused into down's prologue (F5), because
   the prologue fusion forces down to read the 2×-wide gate+up as its input. **Fuse SwiGLU
   into up_gate, not down.**

5. **Two ways to remove the same RMSNorm.** The pre-FFN RMSNorm can fold into the `mla_o`
   epilogue (F2) or into the per-expert up_gate prologue (F3, +×top_k redundant recompute).
   On the FFN (up_gate) side it's a clean −24 MiB save with no SMEM pressure; on the mla_o
   side it rides along with the residual but adds SMEM pressure to that compute-bound GEMM.

6. **The full FFN fusion (F6) is strongest but SMEM-gated; and F1/F2 reveal a starvation
   effect.** F6 eliminates the entire ~384 MiB intermediate and shifts the partition the
   most, but only because the SMEM optimum (~1.3 MiB) is just large enough to hold the
   `m0=64` row-block so both weight matrices are read once (below ~1 MiB it would re-read the
   72 MiB/expert weights and lose — why `ffn_fused` leaves down un-fused). Conversely, F1/F2
   fuse a residual/RMS **epilogue** into the large compute-bound `mla_o` GEMM: the epilogue's
   on-chip output-tile buffer steals SMEM from the GEMM, *raising* its HBM (+336 / +312 MiB),
   though hidden under `mla_o`'s tensor roof. Net: fusing an epilogue into a big compute-bound
   GEMM at tight SMEM can cost HBM — a effect that dominates the prefill study (`SUMMARY_prefill.md`).

## Ranking (HBM saved, min-traffic convention): F6 (−370) > F4 (−256) > F5 (−128) > F3 (−24) | F2 (+312) < F1 (+336) [increase]

The FFN fusions dominate the attention-boundary fusions on HBM, and F4/F6 are the only ones
that change the optimal area split. If the decode layer were not so overwhelmingly
KV-cache-bound (e.g. shorter context, or with sparse/again lower-precision KV), these HBM
savings and the CUDA→tensor rebalance would translate into proportionally larger time wins.

## Reproduce

```
conda run -n fusion python -m fusion.<name>.analysis      # one of:
#   flash_attention_residual, flash_attention_residual_rmsnorm, rmsnorm_up_gate,
#   up_gate_activation, activation_down, up_gate_activation_down
conda run -n fusion python -m fusion.make_summary_figure
```
Each writes plots + a full-grid CSV into `fusion/<name>/result/` (gitignored).
