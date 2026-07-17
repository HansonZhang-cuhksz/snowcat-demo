# Prefill Fusion 3 — pre-FFN RMSNorm + up_gate

Same fused kernel as decode Fusion 3 (`up_gate_rmsnorm`), over the full GLM-5.2 **prefill**
layer (compute/tensor-bound; up_gate M = 32,768 tokens/expert). HBM under the
min-traffic-among-time-optimal convention. See `report.md` (decode).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 3** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12985.384 ms | 1717.681 GiB | 434.753 TFLOP/s |

- **Die split unchanged** (326/858).
- **Time −7.007 ms (−0.054%)** — the removed rmsnorm reduction. **+0.02% redundant
  compute** (+90 GFLOP): the norm is recomputed once per (token, expert) copy (×top_k),
  matching decode; negligible and hidden under the up_gate tensor roof.
- **HBM −12 GiB (saved).** Kernel-level: removed up_gate+rmsnorm 652 GiB → fused 640 GiB.
  The RMS scale is computed inline (aux_hbm=0), so the up_gate GEMM traffic is unchanged and
  the whole saving is the removed rmsnorm's input read (12 GiB = 1,048,576 tokens × 6144 ×
  2 B) — the prefill-scaled analogue of decode's 24 MiB.

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

One of only two prefill fusions that **reduce** HBM (the other is F4), because it fuses a
*reduction* (aux_hbm=0, no SMEM starvation) rather than an output-tile epilogue. It removes
the rmsnorm read (12 GiB) and its time, with no area-split change. As in decode, the
mla_o-side placement (Prefill F2) removes the same RMSNorm while also bundling the
residual, but here (up_gate side) the RMSNorm removal is a clean, if tiny, win.
