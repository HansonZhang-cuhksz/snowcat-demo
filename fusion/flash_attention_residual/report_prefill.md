# Prefill Fusion 1 — FlashAttention (MLA output) + residual add

Same fused kernel as the decode Fusion 1 (`mla_o_residual`), evaluated over the full
GLM-5.2 **prefill** layer (`prefill_area_latency.py` baseline: MLA + DeepSeek Sparse
Attention, K/V materialized, **compute/tensor-bound**; prompt 1,048,576 tokens so every
GEMM's M = tokens). Only the `mla_o`+residual kernel is fused. See `report.md` for the
decode version and `notes/general_fusion_analysis.md` for the shared method.

HBM is reported under the **min-traffic-among-time-optimal** convention (a sane scheduler
minimises traffic when it is free) — required because prefill kernels are compute-bound,
so many tilings tie on time and the raw frontier traffic would be ill-defined.

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 1** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12973.443 ms | 1801.685 GiB | 435.154 TFLOP/s |

- **Die split unchanged** (326 CUDA / 858 tensor). Prefill is tensor-bound; fusing the
  residual doesn't change the tensor GEMM workload, so the optimum doesn't move.
- **Time −18.948 ms (−0.146%)** — the removed residual-add vector kernel. Small (the layer
  is dominated by the DSA lightning-indexer attention core, ~81%).
- **HBM +72 GiB (increase, hidden under compute).** Kernel-level: removed `mla_o`+residual
  272 GiB → fused 344 GiB. `mla_o` is compute-bound (tensor 480.6 ms ≫ mem 181 ms), and the
  fused residual epilogue holds an on-chip output-tile-sized residual buffer (CODA
  convention), stealing SMEM from the huge `mla_o` GEMM (M=1048576, K=16384) → its tiling
  degrades and re-reads more. Because the kernel is tensor-bound, this extra traffic is
  **completely hidden** (no time cost).

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

In prefill the residual fusion is a **pure (small) time win with no area-split change and a
harmless HBM *increase***. Unlike decode (memory-bound, where HBM matters), prefill is
tensor-bound so the fused kernel freely trades HBM for keeping the residual on-chip; the
net effect is just eliminating the residual kernel's compute time. The interesting nuance:
fusing an epilogue into a large compute-bound projection *raises* its HBM (SMEM pressure),
but that is irrelevant to prefill's tensor-bound critical path.
