# Prefill Fusion 4 — up_gate + SwiGLU activation

Same fused kernel as decode Fusion 4 (`up_gate_swiglu`), over the full GLM-5.2 **prefill**
layer (compute/tensor-bound; up_gate M = 32,768/expert). HBM under the
min-traffic-among-time-optimal convention. See `report.md` (decode).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 4** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12917.641 ms | 1601.685 GiB | 437.033 TFLOP/s |

- **Die split unchanged** (326/858) — note this **differs from decode**, where F4 moved the
  split 490→381 CUDA. In decode the removed CUDA-core activation freed area toward tensor;
  in prefill the optimal CUDA allotment (326) is already set by the DSA indexer's
  gate/top-k, so removing the activation doesn't change it.
- **Time −74.750 ms (−0.575%)** — the largest prefill fusion time win (removes the
  activation kernel; FLOPs conserved).
- **HBM −128 GiB (saved)** — the largest prefill HBM win. Kernel-level: removed
  up_gate+activation 736 GiB → fused 608 GiB. Writing only the activated INTERMEDIATE
  columns (pairwise epilogue) instead of the raw gate+up, and eliminating the activation
  round-trip — the prefill-scaled analogue of decode's 256 MiB.

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

The best FFN fusion for prefill: it both **saves the most HBM (128 GiB)** and gives the
largest time win, because the SwiGLU epilogue writes fewer bytes (activated vs raw gate+up)
without stealing GEMM working set (unlike the residual epilogue fusions). Still, the die
split does not move (prefill's CUDA count is indexer-governed), so the fusion's benefit is
traffic + a small time saving, not a partitioning change — the opposite emphasis from
decode, where F4's headline was the CUDA→tensor shift.
