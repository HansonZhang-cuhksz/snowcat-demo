# Step log — Fusion 4: up_gate + SwiGLU activation

- Wrote `plan.md`: fuse SwiGLU into up_gate epilogue via CODA pairwise output tile
  (GEMM produces 2p gate/up cols, epilogue writes p activated cols).
- `model.py`: `up_gate_swiglu` — per-expert M=64, p=INTERMEDIATE=2048 (N=2p=4096), K=6144,
  count=256; tensor = up_gate ops; cuda = SwiGLU (M·p·8); traffic via
  `pairwise_tile_points` with final=m0·p0 (activated), aux=0. Removes `up_gate`, `activation`.
- Ran clean. FLOP conserved exactly (824.902 GFLOP) — no dispatch redundancy (activation
  already per-dispatched-token).
- Result: layer −0.138 ms (−0.011%), **−256 MiB** (matches hand calc: raw gate+up write
  128 + activation 192 − activated write 64). Kernel-level up_gate+activation 6.606 ms /
  12800 MiB → fused 6.448 ms / 12544 MiB.
- **First fusion to move the die split:** 490→381 CUDA, 885→889 tensor (removing the
  CUDA-core activation kernel lowers CUDA demand → area reallocated toward tensor).
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
