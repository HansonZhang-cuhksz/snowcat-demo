# Step log — Fusion 6: up_gate + activation + down (full FFN fusion)

- Checked `ffn_fused_area_latency.py`: it fuses only up_gate+SwiGLU (down stays standard),
  so there was no ready reference for the full GEMM-GEMM fusion — built a custom model.
- Wrote `plan.md`: whole FFN as one kernel; intermediate on chip; SMEM-gated by the
  weight-reread-vs-row-block (`m0`) tradeoff. Documented the register-accumulator ideal
  assumption (weights + x read once per m0-block).
- `model.py` (custom `_points`): enumerate `m0 ∈ divisors(M), ≥16` ({16,32,64}); per m0:
  traffic = x+out (2·M·HIDDEN) + mt·(W_ug+W_dn); buffer = m0·(INTERMEDIATE+HIDDEN)·2 +
  weight/x slices. tensor = up_gate+down; cuda = SwiGLU. Removes up_gate, down, activation.
- Ran clean. FLOP conserved exactly (1237.219 GFLOP).
- Result: layer −0.204 ms (−0.017%), **−384 MiB** (max of all fusions = full intermediate).
  Winning tiling m0=64 (mt=1, weights once), buffer 1.03 MiB, num_stages=1, memory-bound.
  Traffic 18816 MiB = 19200 − 384. Verified by hand.
- **Biggest die shift:** 490→354 CUDA, 885→890 tensor, SMEM 1.316→1.128 MiB — the fusion
  imposes a SMEM floor (≥1.03 MiB to hold m0=64), and the optimizer picks just enough. Kernel
  saving at the fused node is 576 MiB; layer saving 384 (baseline gives up 192 by moving
  to the lower-SMEM design point).
- Key finding: strongest but SMEM-gated — explains why `ffn_fused` leaves down un-fused.
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
