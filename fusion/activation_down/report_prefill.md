# Prefill Fusion 5 — SwiGLU activation + down

Same fused kernel as decode Fusion 5 (`swiglu_down`), over the full GLM-5.2 **prefill**
layer (compute/tensor-bound; down M = 32,768/expert). HBM under the
min-traffic-among-time-optimal convention. See `report.md` (decode).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 5** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12917.641 ms | 1761.685 GiB | 437.033 TFLOP/s |

- **Die split unchanged** (326/858).
- **Time −74.750 ms (−0.575%)** — same as F4 (removes the activation kernel; the down GEMM
  stays tensor-bound so its time is unchanged). FLOPs conserved.
- **HBM +32 GiB (increase, hidden).** Kernel-level: removed activation+down 320 GiB → fused
  352 GiB. The prologue fusion makes down read the **2×-wide gate+up** as its input
  (`widened_a_points`), which at prefill's large M costs more A-read traffic than the
  eliminated activation saves. Hidden under the tensor roof.

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

The prologue-vs-epilogue contrast is sharper in prefill than decode: fusing SwiGLU into
**down's prologue** (F5) *increases* HBM by 32 GiB (the widened gate+up read), while fusing
the same SwiGLU into **up_gate's epilogue** (F4) *saves* 128 GiB. Both give the same time
win (activation removed, hidden under compute) and neither moves the split — but on traffic
F4 dominates F5 by 160 GiB. **Takeaway (reinforcing decode): fuse SwiGLU into up_gate, not
down** — and in prefill the down-side fusion is actively traffic-negative.
