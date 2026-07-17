# Fusion 6 — up_gate + SwiGLU + down (full FFN GEMM-GEMM fusion)

Custom latency-aware roofline+Snowcat analysis of fusing the whole expert FFN
(up_gate → SwiGLU → down) into one kernel, over the full GLM-5.2 decode layer with only
this kernel fused. Baseline = fully-unfused decode layer. See `plan.md` and
`notes/general_fusion_analysis.md`.

## What is fused

`out[M,HIDDEN] = down(SwiGLU(up_gate(x[M,HIDDEN])))` in one kernel; the gate+up and
activated intermediates never touch HBM. Unlike `ffn_fused_area_latency.py` (up_gate+SwiGLU
fused, down standard), this is a **GEMM-GEMM** fusion. Compute conserved exactly
(up_gate + SwiGLU + down = 1237.219 GFLOP fused and removed).

Because down contracts over the full INTERMEDIATE dim, the full `activated[m0,:]` row must
be resident; the kernel processes an `m0`-token row-block and reads **both** weight
matrices once per block (`mt = M/m0` blocks → weights read `mt×`). Modeled by enumerating
`m0`: large `m0` avoids weight re-reads but needs a big on-chip resident; small `m0`
re-reads weights and the fusion loses.

## Results

| | rc | rt | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 1.316 MiB | 490 | 885 | 1224.492 ms | 2326.161 GiB | 246.367 TFLOP/s |
| **Fusion 6** | 0.013 | 0.981 | 1.128 MiB | **354** | **890** | 1224.288 ms | 2325.786 GiB | 246.408 TFLOP/s |

**Layer-level: the largest of all six fusions.** −0.204 ms (−0.0167%), **−384 MiB** HBM =
the entire gate+up + activated round-trip. The **die split shifts the most**: 490→354 CUDA
/ 885→890 tensor, and SMEM settles at **1.128 MiB — just above the 1.03 MiB buffer the
`m0=64` row-block needs**. The fusion imposes a SMEM floor (to hold the resident
intermediate) and the optimizer picks just enough, reallocating the rest away from CUDA.

**Kernel-level** (at the fused optimum, SMEM 1.128 MiB):

| | time | HBM | mapping |
|---|---:|---:|---|
| `up_gate`+`activation`+`down` (unfused) | 10.003 ms | 19392.000 MiB | — |
| `ffn_up_gate_swiglu_down` (fused) | 9.672 ms | 18816.000 MiB | BM=64 BN=6144 BK=2048, M-N-K, num_stages=1 (max_feasible=1), one_stage_smem=1.03 MiB, OI=62.71, BW_eff=2.04 TB/s, **memory-bound** |

The fused kernel picks `m0 = M = 64` (mt=1, both weight matrices read exactly once):
traffic = x(192) + out(192) + W_ug(12288) + W_dn(6144) = **18816 MiB**, i.e. baseline
19200 − 384 (intermediate). At the fused node the unfused stages would cost 19392 MiB
(more, because at 1.128 MiB SMEM the separate up_gate/down tile worse), so the kernel-level
saving there is 576 MiB; netted against the 192 MiB the baseline stages give up by moving
to this lower-SMEM design, the **layer** saving is 384 MiB.

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

The full FFN fusion is the strongest (−384 MiB, all intermediate eliminated) **and** the
most demanding: it only pays off when SMEM holds a large enough row-block (`m0=64` here,
~1 MiB) so both weight matrices are read once. That is precisely why the repo's
`ffn_fused` stops at up_gate+SwiGLU and leaves down standard — a smaller SMEM budget would
force `m0<64`, re-reading the 72 MiB/expert weights `mt×` and making the fusion far worse
than unfused. At the GLM-5.2 decode scale the SMEM optimum (~1.3 MiB, set by the attention
core) happens to be just enough, so the fusion lands its full 384 MiB win and pulls the
die partition furthest toward tensor cores. As with all FFN fusions, the layer stays
KV-cache-bound, so the time win is small (−0.017%) — the value is HBM-traffic and the
CUDA→tensor area rebalance, not the decode-layer critical path.
