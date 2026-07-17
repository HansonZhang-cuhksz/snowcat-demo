# Plan — Fusion 2: FlashAttention + residual add + RMSNorm

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

Baseline attention→FFN boundary (three kernels):
```
mla_o (output-proj GEMM) → y=[BATCH,HIDDEN] to HBM
  → post_attention_residual_add:      h = x_residual + y                 → HBM
  → rmsnorm_square_reduction (pre-FFN RMSNorm): per-row sum-of-squares of h → per-row
      stat to HBM (the 1/rms scale; the actual scale is applied on the next GEMM's read)
```
Fused kernel `mla_o_residual_rmsnorm`: the `mla_o` epilogue adds the residual, applies
the γ weight, and accumulates the per-row sum-of-squares (partial RMS) inline; it writes
the residual-added/γ-weighted hidden `h` plus the per-row RMS partial stat. The raw `y`
is never written/re-read and `h` is never re-read for the reduction. This is CODA's
`gemm_residual_partial_rms_weight` (the "first half" of the GEMM-Residual-RMSNorm-GEMM
reparameterization); the 1/rms scale is applied downstream in the (still-unfused)
up_gate prologue from the partial stats — consistent with the baseline, where RMSNorm
only produces the stat and up_gate reads `h`.

## Fused-kernel model (`model.py`)

Dims: `M=2048, N=HIDDEN=6144, K=N_HEADS·V_HEAD_DIM=16384`, count=1 (same as `mla_o`).

- `tensor_operations = 2·M·N·K` (= mla_o).
- `cuda_operations = residual.operations + rmsnorm.operations`
  (= `POST_ATTENTION_RESIDUAL_ADD_TASK.operations` + `RMSNORM_SQUARE_REDUCTION_TASK.operations`).
- Traffic = CODA `gemm_residual_partial_rms_weight` via `standard_tile_points`:
  - `final_tile_bytes = m0·n0·bpe`                       (write h = D·γ)
  - `aux_hbm_bytes = mt·nt·m0·n0·bpe`  (residual C reads) `+ mt·nt·n0·bpe` (γ reads)
      `+ mt·nt·m0·4`                                    (fp32 partial RMS-stat writes)
  - `aux_buffer_bytes = m0·n0·bpe + n0·bpe + m0·4`
- Tensor-core-feasible tile filter.
- Removes baseline stages: `mla_o`, `post_attention_residual_add`,
  `rmsnorm_square_reduction`.

### Expected traffic delta (hand check, bpe=2, B·H·2 = 24 MiB per tensor)
- Baseline region beyond mla_o GEMM core `G`: y write (24) + residual (read y 24 + read x
  24 + write h 24 = 72) + rmsnorm (read h 24 + tiny stat) = **~120 MiB**.
- Fused: write h (24) + read x (24) + γ + partial stats (small) ≈ **~48 MiB**.
- **Saved ≈ 72 MiB** (drops y-write, y-reread, and h-reread; adds tiny γ/partial-stat
  overhead), i.e. ~24 MiB more than Fusion 1 (the extra is the eliminated RMSNorm
  h-reread). Still negligible vs the 2.3 TiB layer → layer optimum unchanged.

## Analysis (`analysis.py`)
Thin wrapper over `common.run_fusion(model, TITLE, RESULT_DIR, csv)` — baseline swap,
FLOP-conservation check, comparison print, plots, CSV. Same as Fusion 1.

## Verification
- Clean run under `conda run -n fusion python -m fusion.flash_attention_residual_rmsnorm.analysis`.
- FLOP conservation exact (mla_o + residual + rmsnorm).
- Hand-checked ~72 MiB layer saving; fused kernel time ≤ (mla_o+residual+rmsnorm) baseline.
- Note the guarded `INCLUDE_RMSNORM` path (fix from Fusion-1 verification) now matters
  because this fusion removes `rmsnorm_square_reduction`.
