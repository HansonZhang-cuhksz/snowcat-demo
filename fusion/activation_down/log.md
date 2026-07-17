# Step log — Fusion 5: activation + down

- Wrote `plan.md`: SwiGLU fused into down's prologue → down reads gate+up (2×
  INTERMEDIATE) and reduces on chip. Predicted a partial offset (down A doubles).
- Added reusable `widened_a_points(a_width_mult)` to `common.py` (mirrors
  `_traffic_for_standard_output_tile` with a scaled A-tile read/buffer; imports CODA
  `_run_count` / `_partial_accumulator_traffic`).
- `model.py`: `swiglu_down` — per-expert M=64 N=6144 K=2048, count=256; tensor = down ops;
  cuda = SwiGLU (M·K·8); traffic via `widened_a_points(a_width_mult=2)`, final=m0·n0 raw
  down output. Removes `down`, `activation`.
- Ran clean. FLOP conserved exactly (412.585 GFLOP) — down per-dispatched-token, no
  redundancy.
- Result: layer −0.072 ms (−0.006%), **−128 MiB** = 192 (activation) − 64 (down A doubles
  64→128). Kernel activation+down 6592 → fused 6464 MiB. Frontier chose BK=2048 (single
  k-tile, A read once) to minimize widening. Same die shift as Fusion 4 (490→381 CUDA).
- **Key finding: prologue fusion (into down) saves half of epilogue fusion (into up_gate,
  Fusion 4) for the same SwiGLU** — the widened input read is the cost. Fuse SwiGLU into
  up_gate, not down.
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
