# Latency & Pipeline-Stage Model — MoE Decode FFN area analysis

Source of truth: `formula.png` + the Orojenesis snowcat model. Captures the model so
the latency-aware estimators (`ffn_area_latency.py`, `ffn_fused_area_latency.py`) can
be implemented correctly.

## Roofline extension

```
peak = min(compute_roof, OI · BW_eff)
BW_eff = min(BW_physical, (C · N) / latency)      # Little's law: throughput = in-flight bytes / latency
latency = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
```

- `BW_physical` = `bw` (chip HBM bandwidth, scaled by `num_sm`).
- `compute_roof` = tensor-core or CUDA-core roof depending on the stage.

## Variables — CORRECTED (N = W = buffer_bytes)

- `S_total` — total SMEM budget for the area node = `r_smem · A_total / A_bit / 8`.
- `C` — number of concurrent tasks in flight = software-pipeline depth = `num_stages`.
  Per-kernel free variable, integer ≥ 1, no artificial upper bound (the SMEM budget is
  the physical cap).
- `S_eff` — effective SMEM per task = `S_total / C` (fill SMEM: `C·S_eff = S_total`).
- `N = W = buffer_bytes` — **the full one-stage SMEM working set**, including the
  output tile `b` and (for fused stages) the epilogue auxiliary state. The min-traffic
  tiling chosen at capacity `S_eff` has `W(S_eff) ≤ S_eff`. `N` (Little's-law in-flight
  bytes per task) equals `W` — same quantity. (An earlier draft guessed `N` = input
  tiles only; that was wrong — corrected by the user.)
- `OI(S_eff) = ops / min_traffic(S_eff)` — increasing in `S_eff`, diminishing returns.
  Independent of `C` for a fixed tiling (pipelining overlaps the same loads; total
  HBM bytes unchanged). Confirmed.

## The tradeoff with N = W

`BW_eff = C·W/latency = (S_total/S_eff)·W(S_eff)/latency = S_total·(W(S_eff)/S_eff)/latency`.

- `W(S_eff)/S_eff ≈ 1` when `S_eff` is tight (small `S_eff`, large `C`, tiling fills the
  budget) → `BW_eff` near its max `S_total/latency`.
- `W(S_eff)/S_eff < 1` when `S_eff` is loose (large `S_eff`, small `C`, slack) → `BW_eff`
  drops.
- So `BW_eff` decreases with `S_eff`; `OI` increases with `S_eff`. Balance in between.
- Cross-tiling: a large-`W` (high-OI) tiling whose `W` doesn't divide `S_total` leaves
  SMEM slack at fill (`floor(S_total/W)·W < S_total`), so a smaller-`W` tiling that fills
  better can win on `BW_eff` despite lower `OI`.

## Per-(area-node, GEMM) optimization — Option A (Pareto frontier)

Option B (enumerate all tilings, incl. non-Pareto) is a **TODO for a later pass**.
Option A uses the snowcat Pareto frontier (min-traffic tiling at each capacity `= W`).

**Key result:** per-Pareto-point best-C is *exactly* equivalent to the 1-D search over
integer `C`. For a fixed tiling, time is non-increasing in `C` (BW_eff ↑, traffic/OI
constant) and flats once BW saturates, so each frontier point's best `C` is
`min(floor(S_total/W), ceil(BW·latency/W))`, and the min over points recovers the
global optimum. No need to enumerate every integer `C`.

### Per-point evaluation (N = W)

For each frontier point `i` (`W_i = buffer_bytes`, `T_i = traffic`, mapping):

```
C_max_i = floor(S_total / W_i)               # max C with S_eff = S_total/C ≥ W_i (point fits)
if C_max_i < 1: skip                          # tiling doesn't fit even at C=1
C_sat_i = ceil(BW_physical · latency / W_i)   # smallest C saturating physical BW
C_best_i = min(C_max_i, C_sat_i)              # smallest optimal C for this tiling (≥1)
BW_eff_i = min(BW_physical, C_best_i · W_i / latency)
OI_i     = ops / T_i
# standard GEMM (tensor only):
time_i   = count · max(ops / tensor_roof,  T_i / BW_eff_i)
# fused stage (tensor + cuda epilogue):
time_i   = count · max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T_i / BW_eff_i)
```

Min over `i` → stage time; argmin `i` → winning `(tiling, C_best)` for reporting.

Correctness: when `C_sat_i` falls below this tiling's valid `S_eff` range, evaluating at
`C_sat_i` still yields the right time because BW is saturated (time =
`ops/min(roof, OI_i·BW)`, constant across the range) — no underestimate, so the
point-min stays exact.

## Structural change to the latency estimators

- Re-key the frontier on `W = buffer_bytes` (one-stage working set), NOT on
  `buffer_bytes · PIPELINE_NUM_STAGES` as today. This makes the latency frontier
  structurally match the non-latency frontier (`ffn_area.py` style: `buffer_bytes`,
  `traffic_bytes`, `bm/bn/bk/loop_orders`); the `stage_bytes` field becomes redundant
  with `buffer_bytes`.
- Time computation leaves the `searchsorted`-driven `min_traffic_from_frontier` /
  `stage_bytes_from_frontier` path and instead evaluates every frontier point at its
  best `C` per area node (loop over points, vectorize over the area grid; re-run the
  argmin at the best area node for reporting).
- `PIPELINE_NUM_STAGES` is removed from Configuration and from computation; `C` is
  reported per kernel. `HBM_LATENCY_CYCLES` and `CUDA_CLOCK_HZ` stay.

## Scope

Applies to **both** latency versions (`ffn_area_latency.py`,
`ffn_fused_area_latency.py`) only. The two non-latency versions (`ffn_area.py`,
`ffn_fused_area.py`) keep `C=1` semantics (no Little's-law term) and are unchanged.

## Output (at the best area node, SMEM fixed) — per GEMM / GEMM+epilogue kernel

Report `tile (BM,BN,BK)`, `loop_order`, `num_stages = C_best`, plus `one_stage_smem = W`,
`traffic`, `OI`, `BW_eff` for context. (Field set to confirm.)

## Still open

1. Report the **smallest optimal `C = min(floor(S_total/W), ceil(BW·latency/W))`**
   (minimal — don't over-pipeline past BW saturation) vs. the max-feasible
   `C = floor(S_total/W)`?
2. Confirm output field set above.
3. Option B (enumerate all tilings, incl. non-Pareto) stays a TODO for a later pass.
