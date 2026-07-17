# Fusion 2 — FlashAttention (MLA output) + residual add + RMSNorm

Custom latency-aware roofline+Snowcat analysis of fusing the decode MLA output stage
with the post-attention residual add **and** the pre-FFN RMSNorm, over the full GLM-5.2
decode layer with only this kernel fused. Baseline = fully-unfused decode layer
(`decode_area_latency.py`). See `plan.md` and `notes/general_fusion_analysis.md`.

## What is fused

Baseline attention→FFN boundary (three kernels):
```
mla_o (output-proj GEMM, M=2048 N=6144 K=16384) → y=[2048,6144] to HBM
  → post_attention_residual_add:  h = x_residual + y          → HBM
  → rmsnorm_square_reduction:     per-row Σ(h²) → per-row stat → HBM
```
Fused `mla_o_residual_rmsnorm`: the `mla_o` epilogue adds the residual, applies the γ
weight, and accumulates the per-row sum-of-squares inline, writing `h = D·γ` plus the
per-row RMS partial stat. `y` is never written/re-read and `h` is never re-read for the
reduction (CODA `gemm_residual_partial_rms_weight`; the 1/rms scale is applied downstream
in the still-unfused up_gate prologue, exactly as the baseline models it). Compute is
unchanged — tensor `2·M·N·K` + CUDA (residual + RMSNorm) — FLOP conservation exact
(412.355 GFLOP fused vs removed).

HBM under the **min-traffic-among-time-optimal** convention (see `SUMMARY.md`; `mla_o` is
compute-bound).

## Results

| | rc | rt | SMEM | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 1.316 MiB | 885 | 1224.492 ms | 2325.601 GiB | 246.367 TFLOP/s |
| **Fusion 2** | 0.018 | 0.975 | 1.316 MiB | 885 | 1224.443 ms | 2325.906 GiB | 246.377 TFLOP/s |

**Layer-level: null.** −0.049 ms (**−0.0040%**), split unchanged. Total HBM **+312 MiB**
(a 0.01% *increase*, hidden — see below).

**Kernel-level** (at the fused optimum):

| | time | HBM | mapping |
|---|---:|---:|---|
| `mla_o` + `residual` + `rmsnorm` (unfused) | 0.959 ms | 1272.008 MiB | — |
| `mla_o_residual_rmsnorm` (fused) | 0.910 ms | 1584.141 MiB | BM=512 BN=512 BK=16, M-N-K, num_stages=1, OI=248.24, BW_eff=2.04 TB/s, **tensor-bound** |

- Kernel time saved 0.049 ms = the two removed vector kernels (residual 0.037 +
  rmsnorm 0.012); their compute hides under `mla_o`'s tensor roof (0.910 ms).
- Kernel HBM **rises 312 MiB**: the residual + γ + partial-RMS aux tiles steal SMEM from
  the compute-bound `mla_o` GEMM (which standalone gets the full budget → lower traffic),
  degrading its tiling. Hidden under the tensor roof, so time still improves. (This is the
  same SMEM-starvation effect as Fusion 1, slightly larger due to the extra aux.)

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

Adding the RMSNorm to the FlashAttention+residual fusion removes one more vector kernel
than Fusion 1 (−0.049 vs −0.037 ms), still negligible at the decode-layer level (−0.004%)
with the die partition unchanged. The value is removing the residual + RMSNorm kernels at
the attention→FFN boundary; as in Fusion 1, the fused epilogue's on-chip tiles slightly
*raise* `mla_o`'s HBM (SMEM starvation on a compute-bound GEMM), but that is hidden. Not an
area-distribution shift. See `report_prefill.md` for the prefill version.
