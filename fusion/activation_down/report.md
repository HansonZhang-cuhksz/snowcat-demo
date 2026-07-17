# Fusion 5 — SwiGLU activation + down

Custom latency-aware roofline+Snowcat analysis of fusing the SwiGLU activation into the
down GEMM prologue, over the full GLM-5.2 decode layer with only this kernel fused.
Baseline = fully-unfused decode layer. See `plan.md` and `notes/general_fusion_analysis.md`.

## What is fused

```
activation (SwiGLU): reads gate+up [M,4096], writes activated [M,2048] → HBM
down GEMM (per expert M=64 N=6144 K=2048, count=256): reads activated (A), projects
```
Fused `swiglu_down`: the down GEMM reads the gate+up tensor directly, applies SwiGLU on
chip to form the activated A tiles, then contracts over K=INTERMEDIATE. The activated
tensor is never materialized. This is a **prologue** fusion that **widens down's A input
to 2·INTERMEDIATE** (gate+up). Compute conserved exactly (down tensor + SwiGLU CUDA =
412.585 GFLOP fused and removed).

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 1.316 MiB | 490 | 885 | 1224.492 ms | 2326.161 GiB | 246.367 TFLOP/s |
| **Fusion 5** | 0.014 | 0.979 | 1.316 MiB | 381 | 889 | 1224.420 ms | 2326.036 GiB | 246.381 TFLOP/s |

**Layer-level: small.** −0.072 ms (−0.0059%), **−128 MiB** HBM. Same die-split shift as
Fusion 4 (490→381 CUDA, 885→889 tensor) from removing the CUDA-core activation kernel.

**Kernel-level** (at the fused optimum):

| | time | HBM | mapping |
|---|---:|---:|---|
| `activation` + `down` (unfused) | 3.415 ms | 6592.000 MiB | — |
| `swiglu_down` (fused) | 3.323 ms | 6464.000 MiB | BM=64 BN=8 BK=2048, M-N-K, num_stages=2, OI=60.87, BW_eff=2.04 TB/s, **memory-bound** |

Removed HBM 6592 = down 6400 + activation 192; fused 6464 = 6400 + 64 (down's A read
**doubles** from activated 64 MiB to gate+up 128 MiB) with the 192 MiB activation gone.
**Saved 128 MiB.** The frontier picks BK=2048 (full K, single k-tile → A read once) to
minimize the widening cost.

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

Fusing the SwiGLU into **down's prologue** saves 128 MiB — exactly **half of Fusion 4's
256 MiB** for the *same* activation. The difference is structural: a prologue fusion
forces down to read the 2× wider gate+up as its input (+64 MiB), partially offsetting the
192 MiB activation elimination, whereas the epilogue fusion into up_gate (Fusion 4) has
no such cost. **Takeaway: fuse SwiGLU into up_gate (epilogue), not down (prologue).** The
saving stays positive here only because down is weight-dominated so its A-read is small;
had down re-read its input across many N-tiles, the widening could have erased the win.
Like Fusion 4, it slightly rebalances CUDA→tensor area but does not change the KV-cache-
bound layer regime.
