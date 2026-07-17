# Plan — Fusion 1: FlashAttention + residual add

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

Decode MLA output stage. Baseline flow:
```
... MLA attention core → mla_o (output projection GEMM) → y=[BATCH,HIDDEN] to HBM
    → post_attention_residual_add:  out = x_residual + y  → HBM
```
Fused kernel `mla_o_residual`: the `mla_o` GEMM's epilogue adds the residual/skip tensor
`x_residual` to each output tile and writes the sum directly. The raw attention-block
output `y` is never written to / re-read from HBM.

- The attention **core** (`AttentionCoreTask`) stays a separate kernel; the residual is a
  hidden-dim (`[BATCH, HIDDEN]`) op, so its only fuse point is `mla_o` (the kernel that
  produces the hidden-dim block output). This is the faithful 2-op fusion.

## Fused-kernel model (`model.py`)

Dims: `M = padded batch (2048)`, `N = HIDDEN = 6144`, `K = N_HEADS·V_HEAD_DIM = 16384`,
`count = 1`.

- `tensor_operations = 2·M·N·K`  (= baseline `mla_o` ops, 412.317 GFLOP).
- `cuda_operations   = BATCH·HIDDEN·1`  (= residual add, 12.583 MFLOP).
- Traffic points = CODA `gemm_residual` = `_traffic_for_standard_output_tile` per tiling:
  - `final_tile_bytes = m0·n0·bpe`         (write the summed output tile)
  - `aux_hbm_bytes    = mt·nt·m0·n0·bpe`   (read residual C tile once per output tile)
  - `aux_buffer_bytes = m0·n0·bpe`         (residual tile held on chip)
  i.e. `gemm_residual_partial_rms_weight` minus the γ and partial-RMS terms. Reuse
  `coda_fused_traffic._points_over_standard_tiles`.
- Restrict tiles to tensor-core-feasible (`BM≥16, BN≥8, BK≥16`, same as
  `decode_area_latency.tensor_core_tile_allowed`).

### Expected traffic delta (hand check)
Let `G = mla_o GEMM core (a+w+partial) traffic`. Per output-tensor round-trips (bpe=2,
BATCH·HIDDEN·2 = 24 MiB per full tensor):
- Baseline: `mla_o` writes y (24) + residual reads y (24) + reads x (24) + writes sum (24)
  = `G + 96 MiB`.
- Fused: writes sum (24) + reads x (24) = `G + 48 MiB`.
- **Saved ≈ 48 MiB** (the y write + y reread). Against the layer's ~2326 GiB this is
  ~0.002% — so the layer optimum should not move. Report both the (null) layer shift and
  the `mla_o`+residual stage-level shrink.

## Analysis (`analysis.py`)

1. `import decode_area_latency as dal`; `base = dal.evaluate_layer()` → baseline grid,
   roofs, stage times/traffic, `total_time`, `best_index`, `modeled_operations`.
2. Build the fused frontier from `model.py`; compute `fused_time, fused_traffic` over the
   grid via `common.fused_stage_time(frontier, smem_bytes, tensor_roof, cuda_roof)`.
3. `total_fused = base.total_time − task_times["mla_o"] − post_attention_residual_add_time
   + fused_time`;  `best_fused = argmin(total_fused)`.
4. Assert fused FLOPs == removed FLOPs (mla_o tensor + residual cuda). `modeled_operations`
   unchanged.
5. Print comparison: baseline optimum (rc/rt/smem/cores/time/tput) vs fused optimum;
   fused-kernel stage time/traffic/OI/mapping (`num_stages`) vs baseline
   `mla_o`+`residual`; total-HBM-traffic delta at each optimum.
6. Plots into `result/`: (a) fused total-time area map (rt vs rc, log color); (b) baseline
   vs fused total-time-vs-`rt` slice at the baseline `rc`, or a compact 2-panel compare;
   (c) traffic breakdown bar (mla_o + residual vs fused). CSV of the fused sweep.
7. Write `report.md` (follow the repo report style) and append to `log.md`.

## Verification
- Clean run under `conda run -n fusion python -m fusion.flash_attention_residual.analysis`.
- Hand-checked 48 MiB save; fused OI ≈ mla_o OI (traffic barely changes).
- Layer optimum essentially unchanged (documented as the finding); fused kernel stage
  time ≤ mla_o+residual baseline stage time.
