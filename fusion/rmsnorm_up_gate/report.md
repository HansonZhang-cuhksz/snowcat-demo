# Fusion 3 — pre-FFN RMSNorm + up_gate

Custom latency-aware roofline+Snowcat analysis of fusing the pre-FFN RMSNorm into the
up_gate GEMM prologue, over the full GLM-5.2 decode layer with only this kernel fused.
Baseline = fully-unfused decode layer. See `plan.md` and `notes/general_fusion_analysis.md`.

## What is fused

```
rmsnorm_square_reduction: per-row Σ(x²) of the [BATCH,HIDDEN] FFN input → per-row stat → HBM
up_gate GEMM (per expert M=64 N=4096 K=6144, count=256): project dispatched tokens
```
Fused `up_gate_rmsnorm`: up_gate computes the per-row RMS inline while streaming its
K(=HIDDEN) reduction of the A-input it already reads, applying the per-row scale at the
output-tile epilogue (`1/rms` factors out of the K-sum; γ folded into W offline). The
separate rmsnorm kernel is removed. Verified: the CODA plain-GEMM frontier equals the
baseline Snowcat register-accumulator frontier for these dims, so the fused up_gate GEMM
traffic is exactly the baseline's — the only layer traffic change is removing rmsnorm.

**Redundant compute (intentional):** RMSNorm is logically pre-dispatch (2048 tokens) but
up_gate is per-expert (batch·top_k = 16384 rows), so the reduction is recomputed ×top_k.
FLOP accounting: fused 824.835 vs removed 824.659 GFLOP → **+0.176 GFLOP (+0.02%)** — the
7 extra RMS reductions. Negligible CUDA work, hidden under the GEMM's tensor roof.

## Results

| | rc | rt | SMEM | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 1.316 MiB | 885 | 1224.492 ms | 2326.161 GiB | 246.367 TFLOP/s |
| **Fusion 3** | 0.018 | 0.975 | 1.316 MiB | 885 | 1224.480 ms | 2326.138 GiB | 246.369 TFLOP/s |

**Layer-level: null.** −0.012 ms (**−0.0010%**), −24.008 MiB. Die split unchanged.

**Kernel-level** (at the fused optimum):

| | time | HBM | mapping |
|---|---:|---:|---|
| `up_gate` + `rmsnorm` (unfused) | 6.493 ms | 12632.008 MiB | — |
| `up_gate_rmsnorm` (fused) | 6.481 ms | 12608.000 MiB | BM=64 BN=4096 BK=16, M-N-K, num_stages=2 (max_feasible=2), OI=62.39, BW_eff=2.04 TB/s, **memory-bound** |

The fused up_gate is byte-identical to baseline up_gate (6.481 ms / 12608 MiB) — the RMS
reduction adds no HBM (inline) and its compute hides under the memory-bound GEMM. The
whole saving is the removed rmsnorm's 24.008 MiB input read (0.012 ms).

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

Fusing RMSNorm into up_gate removes only the rmsnorm kernel's 24 MiB input read (0.012 ms,
−0.001% of the layer) and adds a negligible ×top_k redundant reduction. It is the
smallest of the attention→FFN-boundary fusions. Notably, **Fusion 2 removes the *same*
pre-FFN RMSNorm more efficiently** — folding it into the mla_o epilogue avoids the
per-expert recompute and bundles the residual+output round-trips (71.9 MiB total vs
24.0 MiB here) — so if the goal is to eliminate the pre-FFN RMSNorm, the mla_o-side
placement dominates the up_gate-side placement. Neither shifts the die partition.
