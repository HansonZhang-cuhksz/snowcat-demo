# Step log — Fusion 1: FlashAttention + residual add

- Read the repo modeling framework: `decode_area_latency.py` (full unfused decode-layer
  baseline: pre-attn RMSNorm → MLA → residual → RMSNorm → MoE FFN → residual, each op a
  separate kernel), `coda_fused_traffic.py` / `coda_fused_register_accumulator_traffic.py`
  (CODA on-chip-intermediate fused traffic), `ski_slope.py` (Snowcat min-traffic),
  `notes/latency_pipeline_model.md` (per-kernel num_stages latency model).
- Confirmed with user: full decode-layer baseline; "FlashAttention" = decode MLA
  flash-decode; CODA on-chip-intermediate traffic model.
- Wrote `notes/general_fusion_analysis.md` (master plan for all 6) and `plan.md` (this
  fusion).
- Built shared engine `fusion/common.py`: `FusedFrontier`, `standard_tile_points` /
  `pairwise_tile_points` (CODA points, register-accumulator loop orders),
  `build_fused_frontier` (Pareto collapse keyed on W), `fused_stage_time` (per-point
  best-C latency), `select_mapping`/`format_mapping`, and the orchestration
  (`swap_fused_kernel`, `print_comparison`, `save_plots`, `write_sweep_csv`). Reuses the
  baseline's chip constants and `decode_area_latency.evaluate_layer()` — no duplication.
- Built `model.py`: fused `mla_o_residual` kernel (mla_o dims + residual epilogue;
  `gemm_residual` = standard tile + residual aux read/buffer); removes baseline stages
  `mla_o`, `post_attention_residual_add`.
- Built `analysis.py`: thin entry point (`python -m fusion.flash_attention_residual.analysis`).
- **Env fix:** the `fusion` conda env is Python 3.10, but `snowcat_demo` used
  `from enum import StrEnum` (3.11+). Added a strictly backward-compatible `StrEnum` shim
  in `snowcat_demo/model/decision.py` (no-op on 3.11+; faithful fallback on 3.10). Only
  existing-file change made.
- Ran clean under `conda run -n fusion`. Baseline optimum reproduces
  `decode_area_report.md` exactly (rc 0.018 / rt 0.975 / 1.316 MiB / 490 / 885 /
  1224.492 ms / 246.367 TFLOP/s) → swap approach validated. FLOP conservation exact.
- Result: layer-level −0.037 ms (−0.0030%), −48 MiB HBM (matches hand calc); kernel-level
  mla_o+residual 0.947 ms/1632 MiB → fused 0.910 ms/1584 MiB. Fused kernel tensor-bound.
- Wrote `report.md`; plots + CSV in `result/` (gitignored).
