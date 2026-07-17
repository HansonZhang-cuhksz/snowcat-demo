# Fusion 4 — up_gate + SwiGLU activation

Custom latency-aware roofline+Snowcat analysis of fusing the SwiGLU activation into the
up_gate GEMM epilogue, over the full GLM-5.2 decode layer with only this kernel fused.
Baseline = fully-unfused decode layer. See `plan.md` and `notes/general_fusion_analysis.md`.

## What is fused

```
up_gate GEMM (per expert M=64 N=4096 K=6144, count=256): writes raw gate+up [M,4096] → HBM
activation (SwiGLU): reads gate+up, writes SiLU(gate)*up [M,2048] → HBM
```
Fused `up_gate_swiglu`: up_gate produces the interleaved gate/up tiles on chip and applies
SwiGLU in the epilogue, writing only the activated [M, INTERMEDIATE] output (CODA pairwise
output tile). The raw gate+up is never written / re-read. Compute conserved exactly
(up_gate tensor + SwiGLU CUDA = 824.902 GFLOP fused and removed).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 1.316 MiB | 490 | 885 | 1224.492 ms | 2326.161 GiB | 246.367 TFLOP/s |
| **Fusion 4** | 0.014 | 0.979 | 1.316 MiB | **381** | **889** | 1224.354 ms | 2325.911 GiB | 246.395 TFLOP/s |

**Layer-level: small but real, and the first fusion to move the die split.** −0.138 ms
(−0.0112%), **−256 MiB** HBM. The optimal partition shifts **490→381 CUDA cores /
885→889 tensor cores**: eliminating the CUDA-core SwiGLU kernel drops CUDA demand, so the
optimizer reallocates ~109 cores' worth of area away from CUDA (toward tensor). Total time
is still KV-cache-bound, so the shift is small — but unlike Fusions 1–3 the partition is
no longer identical to the baseline.

**Kernel-level** (at the fused optimum):

| | time | HBM | mapping |
|---|---:|---:|---|
| `up_gate` + `activation` (unfused) | 6.606 ms | 12800.000 MiB | — |
| `up_gate_swiglu` (fused) | 6.448 ms | 12544.000 MiB | BM=64 BN=4096 BK=16, M-N-K, num_stages=2, OI=62.71, BW_eff=2.04 TB/s, **memory-bound** |

Removed HBM 12800 = up_gate 12608 + activation 192; fused 12544 = 12608 − 64 (activated
write 64 MiB replaces raw gate+up write 128 MiB) with the 192 MiB activation kernel gone.
**Saved 256 MiB** = the raw gate+up write + its reread + the activation output round-trip.

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

up_gate+SwiGLU is the strongest single-pair FFN fusion so far: it eliminates the entire
128 MiB gate/up materialization plus the 192 MiB activation round-trip (256 MiB total,
2× Fusion 1) and — because it removes a **CUDA-core** kernel — it is the first fusion to
nudge the optimal die partition (fewer CUDA cores, more tensor). At the decode layer's
KV-cache-bound scale the time win is still only −0.011%, but the effect on the FFN block
and on the CUDA/tensor balance is the largest of the pairwise fusions.
