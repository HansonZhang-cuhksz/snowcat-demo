# Implementation Plan — Per-Kernel `num_stages` for the Latency Estimators

Goal: make `num_stages` (`C`) a per-kernel optimized mapping parameter in the two
latency-aware estimators, per the model in `notes/latency_pipeline_model.md`. Option A
(Pareto frontier, per-point best-C); Option B (enumerate all tilings) is a TODO.

Scope: `ffn_area_latency.py` and `ffn_fused_area_latency.py` only. The non-latency
`ffn_area.py` / `ffn_fused_area.py` are untouched.

Reference model (see `notes/latency_pipeline_model.md` for full derivation):

```
N = W = buffer_bytes   (full one-stage working set; incl. output tile + fused aux)
BW_eff = min(BW, C·W / latency)            latency = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
C_best = min( floor(S_total/W), ceil(BW·latency / W) )    # smallest optimal C
time_i = count · max(ops/tensor_roof,  T_i/BW_eff)        # standard GEMM
time_i = count · max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T_i/BW_eff)  # fused
```

Per-Pareto-point best-C is exactly equivalent to the 1-D search over integer `C`
(proof in the note). Report **both** `C_best` (minimal-optimal) and `C_max =
floor(S_total/W)` (max-feasible) at the best area node.

## Shared shape of the changes (both files)

### 1. Re-key the frontier on `W = buffer_bytes`

Today the latency frontier is keyed on `buffer_bytes · PIPELINE_NUM_STAGES` (the `C=4`
total). Re-key on `buffer_bytes` (one-stage `W`) so that for a given `S_eff = S_total/C`
we look up "min-traffic tiling with `W ≤ S_eff`". This makes the latency frontier
structurally identical to the non-latency frontier.

- `build_traffic_frontier` (unfused) / `build_fused_frontier` / `build_standard_frontier`
  (fused): change the sort key from `point.buffer_bytes * PIPELINE_NUM_STAGES` to
  `point.buffer_bytes`. Drop the separate `stage_bytes` field (it becomes equal to
  `buffer_bytes`); the dataclass keeps `buffer_bytes`, `traffic_bytes`, `bm`, `bn`, `bk`,
  `loop_orders` (and `stage`/`label`/`count`/`operations` as before). The mapping-tracking
  logic added earlier stays.
- The `improved`-mask / Pareto-collapse logic is unchanged (still
  `frontier_traffic[1:] < frontier_traffic[:-1]`; the fused-latency tie-break on
  `stage_bytes` is dropped since `stage_bytes == buffer_bytes` now — keep the
  traffic-only `improved` mask, matching the non-latency frontier).

### 2. Replace the time functions with a per-point-best-C evaluation

Replace `min_traffic_from_frontier` / `stage_bytes_from_frontier` /
`effective_hbm_bandwidth_from_frontier` usages with a single per-point loop that returns
`(time_array, traffic_array, argmin_point_array)` over the area grid.

**Unfused** — new `gemm_time_from_frontier(frontier, s_total, tensor_roof)`:

```
latency = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
time_best   = full(N, inf); traffic_best = full(N, nan); winner = full(N, -1, int)
for i in range(num_points):
    W_i = frontier.buffer_bytes[i]; T_i = frontier.traffic_bytes[i]
    C_max = floor(s_total / W_i)                      # [N]
    valid = C_max >= 1
    C_sat = ceil(bw * latency / W_i)                  # scalar
    C_best = minimum(C_max, C_sat)
    BW_eff = minimum(bw, C_best * W_i / latency)      # [N]
    mem_time = T_i / BW_eff
    time_i  = frontier.count * maximum(frontier.operations / tensor_roof, mem_time)
    time_i  = where(valid, time_i, inf)
    better = time_i < time_best
    time_best[better]   = time_i[better]
    traffic_best[better]= T_i
    winner[better]      = i
return time_best, traffic_best, winner
```

Vectorize over the ~500k area nodes per point (loop over the ~tens of frontier points).
`N = len(s_total)`. Note: this returns `(time, traffic)` — current `main()` fetches
traffic separately via `min_traffic_from_frontier`; update the call site.

**Fused** — rewrite `fused_stage_time(frontier, s_total, tensor_roof, cuda_roof)` body:
same loop, but `time_i = count * maximum(tensor_ops/tensor_roof, cuda_ops/cuda_roof,
T_i/BW_eff)` with `W_i = frontier.buffer_bytes` (fused working set incl. aux), `T_i =
frontier.traffic_bytes` (fused `hbm_bytes`). Already returns `(time, traffic)` — also
return `winner`.

**Standard (down) in fused file** — rewrite `standard_stage_time(frontier, s_total,
tensor_roof)` body: same as unfused (tensor-only). Already returns `(time, traffic)` —
also return `winner`.

`min_traffic_from_frontier`, `stage_bytes_from_frontier`,
`effective_hbm_bandwidth_from_frontier` are removed (no remaining callers after the
rewrite). Keep `total_hbm_traffic_bytes`.

### 3. Reporting helpers (at the best area node, SMEM fixed)

Replace the current `selected_mapping_from_frontier` / `format_selected_mapping` (which
do a single `searchsorted`) with a per-point argmin that recovers the winning tiling and
its `C`:

```
def select_mapping_from_frontier(frontier, s_total, tensor_roof[, cuda_roof]) -> dict | None:
    # scalar s_total, scalar roofs; loop points; return argmin's:
    # {bm, bn, bk, loop_order, C_best, C_max, W, traffic, OI, BW_eff, time}
```

`format_selected_mapping` prints:
`BM=.., BN=.., BK=.., loop_order=.., num_stages=C_best (max_feasible=C_max),
one_stage_smem=W MiB, traffic=.. MiB, OI=.. FLOP/byte, BW_eff=.. byte/s`

(Unfused takes `tensor_roof`; fused takes `tensor_roof, cuda_roof`. Two thin variants or
one with an optional cuda arg.)

### 4. `main()` updates

- **Unfused**: change
  `weighted_time = weight * gemm_time_from_frontier(...)` +
  `weighted_traffic = weight * frontier.count * min_traffic_from_frontier(...)`
  to
  `t, tr, _ = gemm_time_from_frontier(frontier, smem_bytes, tensor_roof)`
  `weighted_time = weight * t`; `weighted_traffic = weight * frontier.count * tr`.
- **Fused**: `fused_stage_time` / `standard_stage_time` already return `(time, traffic)`;
  extend to capture `winner` if needed for reporting (or re-run `select_mapping_*` at
  `best_index` — simpler, since reporting only needs the winner at one node). **Use the
  re-run approach**: at `best_index`, call `select_mapping_*` with scalar
  `smem_bytes[best_index]` and the scalar roofs to get the winning mapping + C. Avoids
  storing `winner` arrays across the grid.
- Mapping output block (already added in the prior pass): replace
  `format_selected_mapping(stage_frontier, smem_bytes[best_index])` with
  `format_selected_mapping(stage_frontier, smem_bytes[best_index],
  tensor_roof[best_index], [cuda_roof[best_index]])` so it does the per-point argmin and
  prints `num_stages`.
- **Remove** `print(f"Pipeline num_stages: {PIPELINE_NUM_STAGES}")` from Configuration.
  Keep `print(f"HBM latency: {HBM_LATENCY_CYCLES} cycles")` (and the clock line if
  present).
- `PIPELINE_NUM_STAGES` constant: delete (no longer used in computation). Keep
  `HBM_LATENCY_CYCLES`, `CUDA_CLOCK_HZ`.

### 5. Things NOT touched

- `write_csv`, `plot_results`, `make_area_grid`, `total_hbm_traffic_bytes`,
  `_stage_ops_by_name` / `task_operations_by_name`: signatures and inputs unchanged
  (they consume `task_times`/`stage_times`/`task_traffic`/`stage_traffic` arrays, which
  remain `[N]` arrays).
- Non-latency `ffn_area.py` / `ffn_fused_area.py`.
- Frontier *build* multiprocessing (`build_frontiers`).

## Verification

1. Both latency files run clean under `conda activate area`.
2. Numbers move vs. the old `C=4` baseline (expected — both the `C` pin and `N=W` BW
   formula are corrected). Sanity-check direction: with `N=W`, large-`W` high-OI tilings
   that don't divide `S_total` will sometimes lose to smaller-`W` tilings; winning `C`
   often small (1–3) for big GEMMs at small SMEM.
3. Each GEMM stage's mapping line now prints `num_stages=C_best (max_feasible=C_max)`.
4. Random-expert-distribution path (flip `USE_RANDOM_EXPERT_DISTRIBUTION=True` in a temp
   copy) still works: per-aggregate `constituent mappings` each carry their own
   `num_stages`.
5. No `PIPELINE_NUM_STAGES` remains in Configuration output.

## Performance note

Area grid is ~500k nodes (`AREA_GRID_STEP=0.001`). Per-point loop = `num_points × 500k`
per kernel. Default path (3 kernels, ~tens of points each) is fast. Random-expert path
(~70 fused + ~70 standard stages) is heavier (~minutes) but was already non-trivial;
loop-over-points keeps memory at one `~500k` array per pass. If too slow, vectorize the
point axis per kernel (a `[num_points × 500k]` matrix, ~200 MB) — deferred optimization.

## TODO (later pass)

Option B: enumerate all tilings (incl. non-Pareto) and jointly optimize `(tiling, C)`,
since a higher-traffic tiling with more favorable `W` could beat the Pareto point on
`BW_eff`. Requires keeping the raw tiling list instead of the collapsed frontier.

## STATUS: DONE (Option A)

Implemented in `ffn_area_latency.py` and `ffn_fused_area_latency.py`. Both run clean
(`conda run -n area python <file>`, exit 0) in the default path
(`USE_REGISTER_ACCUMULATOR_MAPPINGS=True`, `USE_RANDOM_EXPERT_DISTRIBUTION=False`) and
in the random-distribution path (constituent mappings each carry their own
`num_stages`).

Final best-area-point results (corrected model; numbers move vs. the old `C=4`
baseline as expected — the old pin forced `4·W` SMEM and over-counted in-flight bytes):

| File | rc | rt | SMEM | time | traffic | TFLOP/s |
|---|---|---|---|---|---|---|
| ffn_area_latency.py | 0.018 | 0.970 | 2.257 MiB | 10.613 ms | 20645 MiB | 234.406 |
| ffn_fused_area_latency.py | 0.014 | 0.974 | 2.257 MiB | 10.350 ms | 20133 MiB | 240.382 |

Per-kernel `num_stages` at the best area node (both files): `router`=1, `up_gate`=1
(large tiles already saturate BW at `C=1`), `down`=2 `max_feasible=4` (`C=1` is
latency-bound for the small `down` tile; `C=2` saturates BW). Hand-verified `up_gate`
math (6.58 ms = 256 × max(tensor_time, memory_time)).

`PIPELINE_NUM_STAGES` removed (constant + Configuration line); `HBM_LATENCY_CYCLES`
and `CUDA_CLOCK_HZ` retained. `HBM_CLOCK_HZ` is defined but unused (kept as-is; the
latency clock remains `CUDA_CLOCK_HZ` per the existing code — not changed by this
task).

Note: `ffn_area.py`, `ski_slope.py`, `ski_slope.png` show as modified in git but those
are pre-existing user config edits (BATCH_TOKENS/EXPERTS alignment, DEFAULT_GEMM_MNK),
not part of this task.

