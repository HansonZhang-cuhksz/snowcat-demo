# Why the Snowcat Traffic Model Matters for Design Decisions

Setup: one GLM-5.2 layer on an A100-like chip (2.04 TB/s HBM @ 500 cycles, BF16, area
grid step 0.001). "Snowcat" = Pareto-frontier traffic + per-tiling `num_stages` latency
hiding; "no-snowcat" = algorithmic-minimum traffic + whole-SMEM streaming buffer
(`--no-snowcat`; mechanics in `snowcat_mode_comparison.md`). Attention cores and
vector/norm tasks are identical in both models. Both models re-optimize the rc/rt/SMEM
split unless stated otherwise.

## The two roles of SMEM, and the one term the models disagree on

| SMEM's role | snowcat | no-snowcat |
|---|---|---|
| **Latency hiding (pipelining)** | `C = num_stages` simultaneous copies of the tile working set: `BW_eff = min(bw, C·W/latency)`, `C·W ≤ SMEM` | streaming buffer: `BW_eff = min(bw, SMEM/latency)` |
| **Traffic reduction (reuse)** | `traffic(W)`: Pareto frontier, decreasing in tile size, needs `W ≤ SMEM/C` | **absent** — traffic fixed at the algorithmic minimum |

The pipelining term is implemented differently but measures **identical to within one
tile** in both models (`C·W ≈ bw·latency` — Fact 4). Every divergence below is therefore
the single deleted quantity — **traffic(SMEM)** — entering the critical path.

| Where | Effect of deleting traffic(SMEM) |
|---|---|
| HBM traffic accounting, default chip | understated **4.7×** per prefill layer (2–20× per stage) |
| Time, both models re-optimized, default chip | ≤ 1.3% (invisible) |
| Time prediction for a fixed chip, bandwidth-starved (bw/64) | **1.74×** too fast |
| Silicon allocation, bandwidth-starved (bw/64) | recommended chip **+53.7%** slower |
| Silicon allocation, default chip (0.75 MiB SMEM recommendation) | **+20.0%** slower |
| Silicon allocation, latency-starved (128× latency) | +0.3% (pipelining term dominates → models agree) |

## Fact 1 — The deleted curve: traffic is steep in SMEM

![Traffic amplification vs SMEM](img/snowcat_frontier_amplification.png)

Traffic amplification (snowcat frontier ÷ algorithmic minimum) at given SMEM working sets:

| GEMM | 0.75 MiB | 2.26 MiB | 8.09 MiB | 24.3 MiB | 48.3 MiB | 96 MiB |
|---|---:|---:|---:|---:|---:|---:|
| prefill mla_o (M=1,048,576 × N=6144 × K=16384) | 17.7× | 9.0× | 5.3× | 2.8× | 2.1× | 2.1× |
| prefill up_gate (M=32768 × N=4096 × K=6144) | 9.3× | 4.8× | 3.7× | 1.6× | 1.0× | 1.0× |
| prefill mla_q_b (M=1,048,576 × N=16384 × K=2048) | 8.0× | 4.4× | 1.9× | 1.3× | 1.1× | 1.0× |
| decode up_gate (M=64 × N=4096 × K=6144) | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |

800× at KB-scale SMEM; 2–5× at the 8 MiB design point when all three GEMM dimensions
are large; 1.0× above 0.2 MiB for decode-shaped (small-M, weight-streaming) GEMMs — the
one shape class where the deleted term is genuinely zero.

## Fact 2 — Default design points: the term is off the critical path — times agree, traffic does not

| Analyzer | snowcat time | no-snowcat time | Δtime | snowcat traffic | no-snowcat traffic | Δtraffic |
|---|---:|---:|---:|---:|---:|---:|
| FFN (batch 4096) | 10.613 ms | 10.613 ms | 0.00% | 20,645 MiB | 20,645 MiB | 0% |
| Fused FFN | 10.350 ms | 10.218 ms | −1.3% | 20,133 MiB | 19,877 MiB | −1.3% |
| Decode (batch 2048, 1M ctx) | 1,224.49 ms | 1,224.49 ms | −0.0004% | 2,381,989 MiB | 2,380,105 MiB | −0.08% |
| Prefill (1M prompt, DSA) | 12,992.39 ms | 12,992.39 ms | 0.00% | **4,461,236 MiB** | **943,062 MiB** | **−79%** |
| Inference (prefill + 150 decode) | 13,150.88 ms | 13,150.88 ms | 0.00% | — | — | — |

Two distinct reasons: decode/FFN GEMMs are decode-shaped, so snowcat traffic *is* the
minimum (Fact 1, row 4); prefill carries 4.7× amplification but is tensor-bound, so the
term hides under compute:

![Prefill stage traffic](img/snowcat_prefill_stage_traffic.png)

Every traffic-derived quantity — DRAM energy, bandwidth headroom, multi-SM contention —
is understated 2–20× per stage even where time agrees to the microsecond.

## Fact 3 — Bandwidth scarcity puts the term on the critical path: the models diverge

Prefill, bandwidth scaled down; "designed chip" = no-snowcat's recommended split
evaluated under snowcat physics:

| HBM bw (TB/s) | snowcat optimum | no-snowcat optimum | designed chip | design penalty | fixed-chip prediction error |
|---:|---|---|---:|---:|---:|
| 2.040 | 12,992 ms (8.1 MiB, rt 0.945) | 12,992 ms (8.1 MiB, rt 0.945) | 12,992 ms | +0.0% | 1.00× |
| 1.020 | 13,130 ms (8.1 MiB) | 13,130 ms (8.1 MiB) | 13,130 ms | +0.0% | 1.00× |
| 0.510 | 13,541 ms (8.3 MiB) | 13,520 ms (8.1 MiB) | 13,953 ms | +3.0% | 1.03× |
| 0.255 | 15,720 ms (**24.3 MiB**, rt 0.860) | 14,417 ms (8.1 MiB) | 17,283 ms | +9.9% | 1.20× |
| 0.128 | 19,815 ms (24.3 MiB) | 17,291 ms (8.1 MiB) | 23,993 ms | +21.1% | 1.39× |
| 0.064 | 27,713 ms (**48.3 MiB**, rt 0.734) | 23,793 ms (8.1 MiB) | 37,414 ms | +35.0% | 1.57× |
| 0.032 | 41,807 ms (48.3 MiB) | 37,014 ms (8.1 MiB) | 64,257 ms | **+53.7%** | **1.74×** |

![Bandwidth sweep](img/snowcat_bw_sweep.png)

Snowcat trades cores for SMEM (8 → 24 → 48 MiB, rt 0.945 → 0.734) because larger tiles
cut traffic. No-snowcat's SMEM demand — max(bw·latency buffer, 8.39 MiB attention
working set) — is bandwidth-insensitive (the buffer *shrinks* with bw), so its
recommendation never moves along this axis and the gap grows without bound. The same
divergence appears if tensor throughput rises instead of bandwidth falling
(compute:bandwidth ratio ≳ 8× baseline).

## Fact 4 — Latency scarcity stresses only the pipelining term, which both models share: they converge

FFN (batch 4096), latency scaled up; snowcat's winning up_gate tiling shown:

| HBM latency (cyc) | bw·latency | snowcat optimum | no-snowcat optimum | up_gate tile W | C | C·W | design penalty | fixed-chip pred. error |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 500 | 0.69 MiB | 10.613 ms (2.26 MiB) | 10.613 ms (0.75 MiB) | 1,156 KiB | 1 | 1.13 MiB | **+20.0%** | **1.20×** |
| 2,000 | 2.76 MiB | 10.613 ms (3.57 MiB) | 10.613 ms (2.82 MiB) | 1,156 KiB | 3 | 3.39 MiB | +11.9% | 1.12× |
| 8,000 | 11.04 MiB | 10.615 ms (11.47 MiB) | 10.615 ms (11.09 MiB) | 1,156 KiB | 10 | 11.29 MiB | +4.6% | 1.05× |
| 16,000 | 22.08 MiB | 10.617 ms (22.94 MiB) | 10.617 ms (22.19 MiB) | 1,156 KiB | 20 | 22.58 MiB | +2.1% | 1.02× |
| 32,000 | 44.15 MiB | 10.622 ms (45.51 MiB) | 10.622 ms (44.19 MiB) | 1,156 KiB | 40 | 45.16 MiB | +0.5% | 1.01× |
| 64,000 | 88.31 MiB | 10.879 ms (87.06 MiB) | 10.860 ms (86.50 MiB) | 1,156 KiB | 77 | 86.93 MiB | +0.3% | 1.00× |

(Decode layer, batch 2048/1M ctx: same SMEM trajectory 1.3 → 88 MiB in both models,
times 1,224.5 → 1,233.0 ms, design penalty ≤ +0.1% everywhere — its dominant
KV-stream/attention pipeline is mode-identical by construction.)

![Latency sweep](img/snowcat_latency_sweep.png)

- Snowcat hides latency by replicating the *same* min-traffic tile: W stays 1,156 KiB,
  `C` scales 1 → 77, `C·W` tracks bw·latency to within one tile — SMEM really holds
  `num_stages` copies of the tile footprint.
- Both models' optimal SMEM grows identically with latency (0.75 → 86.5 MiB): for
  latency-driven SMEM sizing the models agree to O(one tile), because the pipelining
  term is mode-independent.
- Divergence *shrinks* as latency grows (+20.0% → +0.3%): once bw·latency ≫ W, any SMEM
  that hides latency automatically fits the min-traffic tile, zeroing the deleted term.
  The largest error is at the **default** 500-cycle point, where no-snowcat's 0.75 MiB
  pick cannot fit the 1.13 MiB min-traffic tile.
- Re-optimized time is latency-insensitive in both models (10.613 → 10.879 ms across
  128×) until the bw·latency footprint displaces cores (87 MiB = 46% of die, rt 0.97 → 0.52).

## Fact 5 — The SMEM axis: only snowcat sees the low-SMEM cliff and the scarce-bandwidth valley

Best achievable time vs SMEM budget (rc/rt re-optimized at every point):

![Time vs SMEM](img/snowcat_time_vs_smem.png)

- Left (FFN, default bw): both curves rise below the ≈0.7 MiB bw·latency buffer (shared
  pipelining term). Between 0.7 and ~2 MiB only snowcat rises — the deleted term: the
  0.75 MiB chip that no-snowcat calls optimal (10.61 ms) really takes 12.73 ms
  (**+20.0%**); 17.7 ms at 0.56 MiB (+67%); 58.9 ms at 0.19 MiB (5.5×).
- Right (prefill, bw/16): no-snowcat's curve is monotonically increasing above the
  8.39 MiB attention floor (its bw·latency buffer is 45 KB here), so it reads extra
  SMEM as pure waste; snowcat's curve has a valley at 24.3 MiB (19.8 s) that the
  no-snowcat design point misses at 24.0 s (+21%).

Feasibility is part of the same deleted machinery: the no-snowcat fused-FFN minimum
keeps the SwiGLU intermediate on chip regardless of size — 512 KB/expert at the decode
default (plausible) but 128 MiB/expert at prefill scale (impossible at any split);
snowcat's working-set check is what rejects such tilings.

## Conclusions

| Question asked of the model | No-snowcat verdict | Requires snowcat? |
|---|---|---|
| Time, compute-bound workload (prefill @ 2 TB/s) | exact (0.00%) | no |
| Time, memory-bound decode-shaped GEMMs, SMEM re-optimized ≥ min-traffic tile | exact to ≤0.2% at every latency tested | no |
| SMEM for latency hiding (pipelining footprint `C·W ≈ bw·latency`) | accurate to O(one tile) at every latency (Fact 4) | no |
| Time, memory-bound with large-operand GEMMs or SMEM below the min-traffic tile | **up to 1.74× under** (Facts 3, 5) | **yes** |
| HBM traffic / DRAM energy / bandwidth headroom | **2–20× per stage under (4.7×/layer)** (Fact 2) | **yes** |
| SMEM for traffic reduction (tile-size choice) | **term absent: +20% penalty at default, +54% at bw/64** (Facts 3–5) | **yes** |
| On-chip feasibility of fusion working sets | unchecked (128 MiB/expert accepted) | **yes** |

One sentence: both models price SMEM's *pipelining* role (bw·latency, num_stages copies
of the tile) the same, so removing snowcat is safe exactly where traffic(SMEM) is flat —
decode-shaped GEMMs or compute-bound layers with SMEM above the min-traffic tile — and
unsafe everywhere a design decision hinges on the traffic(SMEM) curve it deletes: SMEM
sizing near or below the min-traffic tile (+20% at the default chip), core↔SMEM
allocation under bandwidth scarcity (up to +54%), and any traffic/energy accounting (4.7×).
