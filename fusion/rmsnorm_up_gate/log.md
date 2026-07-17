# Step log — Fusion 3: RMSNorm + up_gate

- Wrote `plan.md`: fuse pre-FFN RMSNorm into up_gate prologue (CODA gemm_rms_scale, scale
  computed inline from A reads; γ folded into W). Documented MoE-dispatch redundancy.
- **Empirical check before building:** confirmed the CODA plain-GEMM (aux=0) frontier
  EXACTLY equals the baseline Snowcat register-accumulator frontier for up_gate
  (M=64,N=4096,K=6144): identical 12 points, identical min-traffic at every capacity. So
  using CODA `standard_tile_points` for the fused up_gate is consistent with the baseline
  — no spurious GEMM-traffic delta. (Validates CODA points for all fusions.)
- `model.py`: `up_gate_rmsnorm` — per-expert M=64 N=4096 K=6144, count=256; tensor =
  up_gate ops; cuda = per-expert RMS reduction (M·K + M·(K−1)); traffic via
  `standard_tile_points` with aux_hbm=0, aux_buffer=m0·4 (on-chip RMS accumulator).
  Removes `up_gate`, `rmsnorm_square_reduction`.
- **Bug found + fixed in `common.py`:** `swap_fused_kernel` computed `fused_operations`
  without the `count` factor — wrong for count>1 fusions (showed −821 GFLOP for Fusion 3).
  Fixed to `count·(tensor+cuda)`. Fusions 1/2 (count=1) unaffected.
- Re-ran clean. FLOP accounting: fused 824.835 vs removed 824.659 GFLOP = +0.176 GFLOP
  (+0.02%) redundant = 7×rmsnorm, as designed.
- Result: layer −0.012 ms (−0.001%), −24.008 MiB. Fused up_gate byte-identical to baseline
  up_gate (6.481 ms / 12608 MiB) → only rmsnorm removed. Memory-bound; die split unchanged.
- Key finding: Fusion 2 removes the same RMSNorm more efficiently (mla_o side, no
  per-expert recompute, bundles residual: 71.9 vs 24.0 MiB).
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
