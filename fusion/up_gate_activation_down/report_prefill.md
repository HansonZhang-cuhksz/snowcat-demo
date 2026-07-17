# Prefill Fusion 6 — up_gate + SwiGLU + down (full FFN GEMM-GEMM fusion)

Same fused kernel as decode Fusion 6 (`ffn_up_gate_swiglu_down`), over the full GLM-5.2
**prefill** layer (compute/tensor-bound; **M = 32,768 tokens/expert** — the key difference
from decode's M=64). HBM under the min-traffic-among-time-optimal convention. See
`report.md` (decode).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12992.392 ms | 1729.685 GiB | 434.519 TFLOP/s |
| **Prefill Fusion 6** | 0.012 | 0.945 | 8.086 MiB | 326 | 858 | 12917.641 ms | 2113.685 GiB | 437.033 TFLOP/s |

- **Die split unchanged** (326/858).
- **Time −74.750 ms (−0.575%)** — removes the activation kernel; the two GEMMs stay
  tensor-bound so their time is unchanged. FLOPs conserved.
- **HBM +384 GiB (large increase, hidden).** Kernel-level: removed up_gate+activation+down
  960 GiB → fused **1344 GiB**. This is the **weight-reread catastrophe**: to keep the
  activated intermediate on chip, the fused kernel processes an `m0`-token row-block, but at
  M=32,768 and 8 MiB SMEM the largest feasible block is `m0 ≈ 512` (buffer holds
  `m0·(INTERMEDIATE+HIDDEN)`), so `mt = M/m0 ≈ 64` blocks each re-read **both** weight
  matrices → ~64× the 72 MiB/expert weights. All hidden under the tensor roof.

![area map](result/prefill_total_time_area.png)
![comparison](result/prefill_fusion_comparison.png)

## Conclusion

The decode/prefill contrast is starkest here. In **decode** (M=64) the full FFN fusion is
the *strongest* (−384 MiB, the whole intermediate eliminated, one weight pass). In
**prefill** (M=32,768) the same fusion is **counterproductive**: the row-block can't hold
enough rows on chip, so both weight matrices are re-read ~64× and HBM *increases by
384 GiB*. It only "gets away with it" because prefill is tensor-bound (the extra traffic is
hidden), so the time still ticks down by the removed activation — but there is **no reason
to do this fusion in prefill**. This is the quantitative statement of why full up_gate→down
fusion is a decode-only (small-M) technique, and why `ffn_fused_area_latency.py` leaves down
un-fused for the large-M regimes.
