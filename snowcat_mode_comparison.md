# Snowcat vs. No-Snowcat Traffic-Model Comparison

All area analyzers now support two GEMM traffic models, selected by the module global
`USE_SNOWCAT` (default `True`) or the CLI flag `--no-snowcat`:

| | **Yes-snowcat (default)** | **No-snowcat** |
|---|---|---|
| GEMM HBM traffic | Snowcat/Orojenesis Pareto frontier: minimum traffic achievable with a tiling whose working set fits the SMEM budget | **Algorithmic minimum**: read A (M·K) and B (K·N) once, write C (M·N) once |
| GEMM OI | Depends on SMEM capacity (larger tiles → more reuse) | **Independent of SMEM** (fixed per GEMM shape) |
| Latency hiding (latency analyzers) | Per-tiling `num_stages`: `BW_eff = min(bw, C·W/latency)`, `C = min(floor(SMEM/W), ceil(bw·latency/W))` | Whole SMEM as one ideal streaming buffer: `BW_eff = min(bw, SMEM_total/latency)` |
| Character | Achievable by a real tiled kernel | **Overly optimistic** (perfect on-chip reuse of every operand) |

Unchanged in both modes: the attention cores (decode KV streaming, prefill flash/DSA), the
vector/norm/reduction tasks (their traffic is already analytic operands+results), the
even-routing expert split, and the M=16 padding of small per-expert GEMMs (kept so the two
modes differ **only** in the traffic model). In the fused analyzers, no-snowcat additionally
keeps the fused chain's intermediates on chip: `up_gate_rms_swiglu` reads input+weights only
and `down` reads weights + writes output only (the SwiGLU activations never touch HBM).

```text
Run any analyzer with:  python <analyzer>.py --no-snowcat
Outputs get a "no_snowcat" filename suffix (e.g. ffn_area_latency_register_accumulator_no_snowcat_times.csv),
so default-mode results are never overwritten.  inference/batched propagate the flag to both stage models.
```

## Headline results (optimal area split, both modes)

| Analyzer | Mode | rc | rt | SMEM (MiB) | CUDA | Tensor | Time | Total HBM (MiB) | TFLOP/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ffn_area | snowcat | 0.018 | 0.970 | 2.257 | 490 | 880 | 10.6131 ms | 20,645 | 234.41 |
| ffn_area | **no-snowcat** | 0.018 | 0.981 | **0.188** | 490 | 890 | 10.6128 ms | 20,645 | 234.41 |
| ffn_area_latency | snowcat | 0.018 | 0.970 | 2.257 | 490 | 880 | 10.6131 ms | 20,645 | 234.41 |
| ffn_area_latency | **no-snowcat** | 0.018 | 0.978 | **0.752** | 490 | 888 | 10.6128 ms | 20,645 | 234.41 |
| ffn_fused_area | snowcat | 0.014 | 0.974 | 2.257 | 381 | 884 | 10.3498 ms | 20,133 | 240.38 |
| ffn_fused_area | **no-snowcat** | 0.014 | 0.985 | **0.188** | 381 | 894 | **10.2179 ms** | 19,877 | 243.49 |
| ffn_fused_area_latency | snowcat | 0.014 | 0.974 | 2.257 | 381 | 884 | 10.3498 ms | 20,133 | 240.38 |
| ffn_fused_area_latency | **no-snowcat** | 0.014 | 0.982 | **0.752** | 381 | 891 | **10.2180 ms** | 19,877 | 243.48 |
| decode (batch 2048, 1M ctx) | snowcat | 0.018 | 0.975 | 1.316 | 490 | 885 | 1,224.492 ms | 2,381,989 | 246.37 |
| decode (batch 2048, 1M ctx) | **no-snowcat** | 0.018 | 0.978 | **0.752** | 490 | 888 | 1,224.487 ms | 2,380,105 | 246.37 |
| prefill (1M prompt, DSA) | snowcat | 0.012 | 0.945 | 8.086 | 326 | 858 | 12,992.392 ms | 4,461,236 | 434.52 |
| prefill (1M prompt, DSA) | **no-snowcat** | 0.012 | 0.945 | 8.086 | 326 | 858 | 12,992.392 ms | **943,062** | 434.52 |
| inference (prefill + 150 decode) | snowcat | 0.012 | 0.945 | 8.086 | 326 | 858 | 13,150.879 ms | — | 431.12 |
| inference (prefill + 150 decode) | **no-snowcat** | 0.012 | 0.945 | 8.086 | 326 | 858 | 13,150.879 ms | — | 431.12 |
| batched inference N=1 / N=8 | snowcat | 0.012 | 0.945 | 8.086 | 326 | 858 | 13,150.9 / 105,033.0 ms | — | — |
| batched inference N=1 / N=8 | **no-snowcat** | 0.012 | 0.945 | 8.086 | 326 | 858 | 13,150.9 / 105,033.0 ms | — | — |

**Removing Snowcat changes execution time by at most 1.3% (fused FFN) and usually by ~0%; what
it changes is the SMEM requirement and the reported traffic/OI.**

## Why the times barely move

1. **Decode-shaped GEMMs: Snowcat already achieves the algorithmic minimum.** The MoE
   up_gate/down and expert-count-M GEMMs are weight-dominated (K·N ≫ M·K, M·N). Even at
   ~1–2 MiB SMEM the Snowcat frontier's best tiling streams the weights once, so the two
   models produce **identical traffic** (up_gate 12,928 MiB, down 6,656 MiB in both modes)
   and identical bandwidth-bound times.
2. **Prefill-shaped GEMMs: traffic differs hugely but is off the critical path.** At the
   prefill optimum the layer is tensor-throughput bound, so GEMM time = ops/tensor_roof in
   both modes → identical times even though traffic drops 4.7×.
3. **The dominant memory streams are not GEMMs.** Decode's 2.3 TiB KV-cache read and the
   vector/norm traffic are analytic in both modes.

## What does change

### 1. SMEM shrinks to the bandwidth-delay product
With OI decoupled from SMEM, the only remaining use of SMEM (latency analyzers) is the
streaming buffer `BW_eff = min(bw, SMEM/latency)`. Saturating `bw = 2.04 TB/s` at 500 cycles
(354.6 ns) needs `bw·latency ≈ 723 KB`; the optimizer picks the smallest grid node above it,
**0.752 MiB** (vs 1.3–2.3 MiB under Snowcat). Non-latency analyzers keep no use for SMEM at
all and collapse to the grid minimum (0.188 MiB). The freed area buys ~3–10 extra tensor
cores — worth ≤0.4% in time.

Two caveats (measured in `snowcat_importance_report.md`, Facts 4–5): the bw·latency
pipelining footprint is not Snowcat-specific — under Snowcat it appears as
`num_stages` simultaneous copies of the tile working set (`C·W ≈ bw·latency`), so both
models' SMEM demand grows identically with latency (0.75 → 86.5 MiB at 500 → 64,000
cycles). And the smaller no-snowcat recommendation is not free: at 0.752 MiB the
Snowcat-modeled FFN runs +20% slower (12.73 vs 10.61 ms) because the 1.13 MiB
min-traffic up_gate tile no longer fits — the mode difference shows up in the *design*,
not in the re-optimized time.

**Exception — prefill (and therefore inference/batched) is bit-identical in both modes**: its
optimum is pinned at 8.086 MiB not by Snowcat but by the flash-attention working set
(`ATTN_FLASH_BLOCK = 128` rows × 64 heads × 512 × 2 B = 8.39 MiB per pipeline stage, one stage
minimum). That constraint is part of the attention model, which no-snowcat leaves untouched,
so the entire prefill-governed family reports the same split, same time, same throughput.

### 2. Reported traffic/OI becomes (much) more optimistic — prefill worst case
Per-stage prefill comparison (times identical, all tensor-bound):

| Stage | Snowcat traffic | No-snowcat traffic | ratio | Snowcat OI | No-snowcat OI |
|---|---:|---:|---:|---:|---:|
| mla_q_a | 100.0 GiB | 16.0 GiB | 6.2× | 245.8 | 1,533.8 |
| mla_q_b | 288.0 GiB | 36.1 GiB | 8.0× | 227.6 | 1,817.3 |
| mla_kv_a | 26.6 GiB | 13.1 GiB | 2.0× | 259.6 | 526.4 |
| mla_kv_b | 120.0 GiB | 57.0 GiB | 2.1× | 238.9 | 502.8 |
| mla_o | 908.0 GiB | 44.2 GiB | 20.5× | 216.5 | 4,449.4 |
| router | 14.0 GiB | 12.5 GiB | 1.1× | 219.4 | 245.7 |
| up_gate | 1,600.0 GiB | 172.0 GiB | 9.3× | 245.8 | 2,286.1 |
| down | 864.0 GiB | 134.0 GiB | 6.4× | 227.6 | 1,467.2 |
| **layer total** | **4,357 GiB** | **921 GiB** | **4.7×** | | |

At the pinned 8 MiB SMEM, Snowcat's tilings of the huge-M (up to 8.4M-row) prefill GEMMs can
only reuse a tile-sized slice of each operand, so real traffic is 2–20× the algorithmic
minimum. No-snowcat hides this amplification — that is precisely the "overly optimistic"
caveat: it under-reports HBM traffic (and thus energy/DRAM pressure) by ~4.7× for prefill
even though the time is unaffected.

Decode per-stage: batch-M GEMMs also drop (mla_o 1,560→280 MiB, mla_q_b 576→136 MiB) but the
expert GEMMs and the absorption GEMMs are already at the minimum; layer traffic falls only
0.08% (KV cache dominates), time −0.0004%.

### 3. Fused FFN gets its intermediates for free
No-snowcat's fused minimum keeps the SwiGLU output on chip: up_gate 12,800→12,672 MiB (−128
MiB write) and down 6,656→6,528 MiB (−128 MiB read), i.e. the 256 MiB round-trip of the
M×2048 intermediates disappears. Both stages are memory-bound, so the fused layer speeds up
10.350→10.218 ms (**−1.3%**, the largest time change in any analyzer). The unfused FFN is
unchanged (its per-GEMM minimum still includes the intermediate write/read), so the
fused-vs-unfused gap widens from 2.5% to 3.7% under no-snowcat.

## Conclusion

The no-snowcat mode confirms that the **area conclusions are robust to the traffic model**:
decode stays bandwidth-bound and split-indifferent, prefill stays tensor-bound and governs the
combined design (same 858-tensor/326-CUDA split, pinned by the 8 MiB flash-attention tile),
and end-to-end inference time is identical to the digit. What Snowcat actually contributes at
these optima is not time but **fidelity of the memory-system picture**: the honest SMEM
requirement (enough for the min-traffic tile *times* the `num_stages ≈ bw·latency/W`
in-flight copies, rather than a bare bw·latency streaming buffer) and the real HBM traffic,
which the algorithmic-minimum model understates by up to 4.7× (prefill) — the gap a real
chip would pay in DRAM energy and bandwidth headroom. Use `--no-snowcat` as an optimistic lower bound on
traffic and an upper bound on how much any smarter tiling could ever help: for these
workloads, at most 1.3% of time.
