# Plan — Fusion 3: RMSNorm + up_gate

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

The pre-FFN RMSNorm fused into the up_gate GEMM **prologue** (instead of the mla_o
epilogue of Fusion 2). Baseline:
```
rmsnorm_square_reduction: per-row Σ(x²) of the [BATCH,HIDDEN] FFN input → per-row stat → HBM
up_gate GEMM (per expert, M=64 N=4096 K=6144, count=256): reads dispatched tokens x, projects
```
Fused `up_gate_rmsnorm`: up_gate computes the per-row RMS inline while streaming its K
(= HIDDEN) reduction of the A-input it already reads, and applies the per-row scale at
the output-tile epilogue. Because RMSNorm's `1/rms(x[m,:])` is a per-row scalar, it
factors out of the K-sum: `O[m,n] = (1/rms(x[m,:])) · Σ_k (x[m,k]·γ[k])·W[k,n]` — so γ
is folded into W offline and the scale is applied at the epilogue (CODA `gemm_rms_scale`,
but with the scale computed **inline** from the A reads, not read from HBM). The separate
rmsnorm kernel (which reads the [BATCH,HIDDEN] tensor) is removed.

### MoE-dispatch note (redundant compute — intentional)
RMSNorm is logically pre-dispatch (2048 unique tokens); up_gate is per-expert on the
dispatched tokens (batch·top_k = 16384 rows). Fusing the reduction into per-expert
up_gate recomputes the norm once per (token, expert) copy → **×top_k = 8 redundant RMS
reductions**. This is negligible: RMSNorm is 25.16 MFLOP vs up_gate's 824.6 GFLOP, so 8×
adds ~176 MFLOP (~0.02% of up_gate), and it is CUDA work that hides under the GEMM's
tensor roof. FLOPs are therefore intentionally **not** conserved; the report states the
+7×rmsnorm compute delta explicitly.

## Fused-kernel model (`model.py`)

Per-expert dims: `M=64, N=2·INTERMEDIATE=4096, K=HIDDEN=6144`, count=256.

- `tensor_operations = 2·M·N·K` (per expert; = baseline up_gate).
- `cuda_operations = M·K + M·(K−1)` (per expert; the RMS square-reduction over K, matching
  `ReductionTask` op accounting). Total over experts = ×top_k baseline rmsnorm.
- Traffic via `standard_tile_points`:
  - `final_tile_bytes = m0·n0·bpe`     (write raw gate+up output; activation is separate)
  - `aux_hbm_bytes = 0`                (scale computed inline; γ folded into W offline)
  - `aux_buffer_bytes = m0·4`          (on-chip fp32 per-row RMS accumulator)
- Verified: CODA plain-GEMM (aux=0) frontier == baseline Snowcat register-accumulator
  frontier for these dims (identical points + min-traffic at all capacities), so the
  fused up_gate GEMM traffic is exactly consistent with the baseline; the only layer-level
  traffic change is removing rmsnorm.
- Removes baseline stages: `up_gate` (GEMM), `rmsnorm_square_reduction` (reduction).

### Expected delta
Layer saving = the removed rmsnorm kernel's HBM = **24.008 MiB** (read [BATCH,HIDDEN] +
tiny stat write). up_gate GEMM traffic/time unchanged. Layer time saving = rmsnorm time
(~0.012 ms). Negligible; layer optimum unchanged.

## Verification
- Clean run under `conda run -n fusion python -m fusion.rmsnorm_up_gate.analysis`.
- up_gate stage traffic in the fused == baseline up_gate (kernel-level delta = rmsnorm only).
- FLOP accounting shows +7×rmsnorm (~176 MFLOP) redundant compute, as designed.
