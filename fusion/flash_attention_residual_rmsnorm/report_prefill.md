# Prefill Fusion 2 — FlashAttention (MLA output) + residual + RMSNorm

Same fused kernel as decode Fusion 2 (`mla_o_residual_rmsnorm`), over the full GLM-5.2
**prefill** layer (compute/tensor-bound; every GEMM M = 1,048,576 tokens). HBM under the
min-traffic-among-time-optimal convention. See `report.md` (decode) and
`notes/general_fusion_analysis.md`.

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 2** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12966.436 ms | 1789.709 GiB | 435.389 TFLOP/s |

- **Die split unchanged** (326/858) — tensor-bound.
- **Time −25.956 ms (−0.200%)** — the removed residual-add + RMSNorm vector/reduction
  kernels. FLOPs conserved (mla_o + residual + rmsnorm).
- **HBM +60 GiB (increase, hidden).** Kernel-level: removed 284 GiB → fused 344 GiB. Same
  `mla_o` SMEM-starvation as Fusion 1 (the residual + γ + partial-RMS aux tiles steal
  working set from the compute-bound GEMM), plus the RMSNorm reduction now piggybacks on
  the GEMM. All hidden under the tensor roof.

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

Like Prefill Fusion 1, this is a small time win (removing two more vector kernels, so
slightly larger than F1's −18.9 ms) with **no area-split change** and a harmless HBM
increase. Folding the residual **and** the pre-FFN RMSNorm into `mla_o` removes both
vector kernels from the (cheap, CUDA-side) critical path; the added GEMM SMEM pressure is
irrelevant because prefill is tensor-bound.
