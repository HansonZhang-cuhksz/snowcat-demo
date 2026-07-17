# Fusion 1 — FlashAttention (MLA output) + residual add

Custom latency-aware roofline+Snowcat analysis of fusing the decode MLA output stage
with the post-attention residual add, evaluated over the **full GLM-5.2 decode layer**
with only this kernel fused. Baseline = fully-unfused decode layer
(`decode_area_latency.py`). See `plan.md` and `notes/general_fusion_analysis.md`.

## Assumptions

Identical chip/workload constants to the decode baseline (single SM: `A_total`=136.29 M
µm², SRAM 0.0864 µm²/bit, HBM 2.04 TB/s @ 500-cycle latency, tensor 512 GFLOP/s/core,
CUDA 5.64 GFLOP/s/core, 1410 MHz; GLM-5.2 MLA, batch 2048, KV context 1,048,576, 256
experts top-8, BF16). Register-accumulator loop orders; even expert routing.

## What is fused

Baseline attention output path (two kernels):
```
MLA core → mla_o (output-proj GEMM, M=2048 N=6144 K=16384) → y=[2048,6144] to HBM
         → post_attention_residual_add:  out = x_residual + y  → HBM
```
Fused `mla_o_residual`: the `mla_o` epilogue adds the residual tile and writes the sum;
the raw output `y` is never written / re-read (CODA on-chip intermediate). Compute is
unchanged: tensor = `2·M·N·K` = 412.317 GFLOP, CUDA epilogue = `BATCH·HIDDEN` =
12.583 MFLOP. FLOP conservation verified exactly (412.329 GFLOP fused vs removed).

Traffic (register-accumulator, K innermost, no partial spill):
`GEMM core reads + write summed tile (m0·n0) + read residual tile (m0·n0)`. Versus
baseline (`mla_o` writes y + residual reads y + reads x + writes sum), fusion drops the
`y` write and `y` re-read = **48 MiB** (2 × BATCH·HIDDEN·2 B).

HBM is reported under the **min-traffic-among-time-optimal** convention (see
`SUMMARY.md`); `mla_o` is compute-bound, so this matters here.

## Results

| | rc | rt | r_smem | SMEM | CUDA | Tensor | Total time | Total HBM | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused baseline | 0.018 | 0.975 | 0.007 | 1.316 MiB | 490 | 885 | 1224.492 ms | 2325.601 GiB | 246.367 TFLOP/s |
| **Fusion 1** | 0.018 | 0.975 | 0.007 | 1.316 MiB | 490 | 885 | 1224.455 ms | 2325.929 GiB | 246.374 TFLOP/s |

**Layer-level effect: essentially null.** Total time −0.037 ms (**−0.0030%**), split
unchanged. Total HBM **+336 MiB** (a tiny 0.01% *increase* — see below), invisible against
the layer's 2.3 TiB, which is ~99% MLA-attention-core (KV-cache) bound.

**Kernel-level effect** (at the fused optimum):

| | time | HBM | mapping |
|---|---:|---:|---|
| `mla_o` + `residual_add` (unfused) | 0.947 ms | 1248 MiB | — |
| `mla_o_residual` (fused) | 0.910 ms | 1584 MiB | BM=512 BN=512 BK=16, M-N-K, num_stages=1 (max_feasible=1), OI=248.25, BW_eff=2.04 TB/s, **tensor-bound** |

The fused kernel saves the full residual-add **time** (0.037 ms) — the residual's compute
hides under `mla_o`'s tensor roof (0.910 ms). But its **HBM rises by 336 MiB**: the
residual epilogue holds an on-chip output-tile-sized residual buffer (CODA convention),
which at the tight 1.316 MiB SMEM steals working set from the large compute-bound `mla_o`
GEMM (K=16384), degrading its tiling so it re-reads more than the standalone `mla_o` (which
gets the full SMEM). Since `mla_o` is tensor-bound, that extra traffic is **completely
hidden** — the fusion is still a (tiny) net win on time.

![area map](result/total_time_area.png)
![comparison](result/fusion_comparison.png)

## Conclusion

Fusing FlashAttention (MLA output projection) with the residual add is a negligible
win at the decode-layer level (−0.003% time) because the layer is overwhelmingly
KV-cache-bandwidth bound. Its value is removing the residual as a separate kernel (its
compute folds under `mla_o`'s tensor roof); the subtlety is that the residual epilogue's
on-chip tile actually *raises* `mla_o`'s HBM slightly (SMEM pressure on a compute-bound
GEMM), but that traffic is hidden. It does **not** change how the die should be partitioned
(still tensor-heavy, rt≈0.975, ~1.3 MiB SMEM). See `report_prefill.md` for the prefill
version (same effect, larger absolute numbers, still hidden — prefill is tensor-bound).
