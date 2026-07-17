# Decode Batch-Size Sweep — effect on fusion optimality

**Question:** at the default GPU spec (2.04 TB/s, 500 cyc, 512 / 5.64 GFLOP/s tensor/CUDA)
and default KV context (1,048,576), how does the **decode batch size** change (a) how much
each fusion helps and (b) *which* fusion is optimal?

Method: `fusion/batch_sweep.py`. Sweep batch ∈ {32 … 32768} (powers of two), and at each
batch re-`configure()` the decode workload (which sets `TOKENS_PER_EXPERT = batch·top_k/256`
and every GEMM's M), rebuild the Snowcat frontiers, and optimise the die split for the
unfused baseline and each of the six fusions. Batch < 32 is excluded (there `batch·8/256 < 1`
so not all 256 experts receive a token — the reduced-active-expert regime, which the fusion
models' `count = EXPERTS` does not represent). Area grid 0.002 → split shift quantised to
~55 cores (trend-level). Everything is compared to the default batch **2048**.

![batch sweep](result/batch_sweep.png)

## Context — attention dominance grows with batch

The FFN weight-streaming GEMMs (up_gate/down read all 256 experts' weights) are **batch-
independent**, while the MLA attention core scales with batch (each sequence carries its own
1M-token KV cache). So the attention time fraction climbs with batch:

| batch | 32 | 64 | 128 | 256 | 512 | 2048* | 8192 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens/expert | 1 | 2 | 4 | 8 | 16 | 64 | 256 | 1024 |
| attention time % | 66.1 | 79.6 | 88.6 | 93.9 | 96.8 | 99.0 | 99.6 | 99.6 |

The FFN — where every fusion except F1/F2 acts — is a real slice only at **small batch**.

## Finding 1 — Fusion's time benefit shrinks as batch grows

Best-fusion Δtime: **+0.091% at batch 32 → +0.010% at batch 32768** (~9× smaller). As
attention takes over the layer, the FFN fusions have less to work on. (Note all magnitudes
are < 0.1% — decode is attention/KV-cache bound even at batch 32, so fusion is never a large
*time* lever; the batch dependence is about relative benefit and fusion *choice*.)

## Finding 2 — The optimal fusion CHANGES with batch (the headline)

| batch (tokens/expert) | best-time fusion | why |
|---|---|---|
| 32 – 4096  (≤128) | **F6 up_gate+act+down** | full-FFN fusion; per-expert M small, so the on-chip row-block holds all rows and both weight matrices are read once → the whole intermediate is eliminated cheaply |
| **8192 – 32768 (≥256)** | **F4 up_gate+act** | F6 loses: at M/expert ≥ 256 the row-block no longer fits at ~1.3 MiB SMEM (`m0 ≤ ~83`), so F6 re-reads both weight matrices `mt = M/m0 ≈ 4×`; F4 (epilogue SwiGLU, no weight re-read) overtakes |

The **F6 → F4 crossover at batch ≈ 8192** is the same SMEM-gated weight-reread effect that
makes F6 catastrophic in prefill (M = 32768) — here it is driven *within decode* by batch:
growing the batch grows tokens/expert, pushing the FFN toward the large-M regime where the
full up_gate→down fusion stops paying off. **So the "best fusion" is batch-dependent: fuse
the whole FFN (F6) at small batch; stop at up_gate+SwiGLU (F4) at large batch.**

## Finding 3 — F5 (act+down) is counterproductive at small batch

F5 (SwiGLU fused into down's prologue) is **negative at small batch** (−0.02% at batch 32,
−0.01% at 64): the down-prologue reads the 2×-wide gate+up, and that widening penalty
exceeds the activation saving when `down` is a larger share of a small-batch FFN. It turns
marginally positive (~+0.01%) only for batch ≥ 256. (Reinforces the epilogue-beats-prologue
rule: F4 into up_gate always ≥ 0; F5 into down can lose.)

## Finding 4 — The die-split impact is batch-robust

The activation-removing fusions (F4/F5/F6) shift the optimal CUDA-core count by **~110–164
cores at every batch** (the ~55-core steps are grid quantization); the attention-boundary
fusions (F1/F2/F3) never move it, at any batch. So while the *magnitude* of the time win and
the *choice* of best fusion vary with batch, the *fact* that fusing the CUDA-side SwiGLU
rebalances the die (freeing CUDA area toward tensor) holds across the whole 1000× batch range.

## Bottom line

- **Which fusion is optimal depends on batch:** F6 (full FFN) at small/mid batch
  (tokens/expert ≤ 128), **F4 (up_gate+SwiGLU) at large batch** (≥ 256) once F6's weight
  re-reads bite. Crossover ≈ batch 8192.
- **How much fusion helps (time) shrinks with batch** — from ~0.09% (batch 32) to ~0.01%
  (batch 32768) as attention dominates — but is always small (decode is attention-bound).
- **Whether fusion moves the die split does not depend on batch** — the activation-removing
  fusions always shift it ~110–164 CUDA cores; attention fusions never do.
- **Practical rule:** small decode batches are where fusion (and specifically the full-FFN
  fusion F6) is most worth doing; as batch grows, downgrade to F4 and expect ever-smaller
  time gains, though the CUDA→tensor area rebalance remains.

## Reproduce
```
conda run -n fusion python -m fusion.batch_sweep         # writes result/batch_sweep.json
conda run -n fusion python -m fusion.make_batch_figure
```
