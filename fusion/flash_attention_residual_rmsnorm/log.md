# Step log — Fusion 2: FlashAttention + residual add + RMSNorm

- Reused the shared engine `fusion/common.py` (built + verified during Fusion 1; 0
  confirmed defects from the adversarial-verification workflow). Applied the
  `INCLUDE_RMSNORM` guard fix to `_vector_stage_traffic()` (surfaced by that workflow) —
  now exercised here because this fusion removes `rmsnorm_square_reduction`.
- Wrote `plan.md`: fuse `mla_o` + residual + pre-FFN RMSNorm as CODA
  `gemm_residual_partial_rms_weight` (first half of the GEMM-Residual-RMSNorm-GEMM
  reparameterization).
- `model.py`: `mla_o_residual_rmsnorm` — same dims as mla_o; tensor = 2·M·N·K; cuda =
  residual + rmsnorm ops; traffic via `standard_tile_points` with residual + γ + partial
  fp32 RMS-stat aux. Removes `mla_o`, `post_attention_residual_add`,
  `rmsnorm_square_reduction`.
- `analysis.py`: thin wrapper over `common.run_fusion`.
- Ran clean under `conda run -n fusion`. Baseline optimum reproduces the decode baseline
  exactly. FLOP conservation exact (412.355 GFLOP).
- Numeric self-consistency confirmed: removed HBM 1656.008 = mla_o 1560 + residual 72 +
  rmsnorm 24.008; fused 1584.141 = 1584 + γ 0.047 + partial-RMS 0.094 (BM=512/BN=512,
  mt=4 nt=12) → saved 71.867 MiB, matching the ~72 MiB hand estimate.
- Result: layer-level −0.049 ms (−0.004%), −71.867 MiB (≈ Fusion 1 + the RMSNorm
  h-reread). Fused kernel tensor-bound; kernel time saved 0.049 ms = residual + rmsnorm.
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
