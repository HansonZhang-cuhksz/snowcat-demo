# Plan — Fusion 5: activation (SwiGLU) + down

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

```
activation (SwiGLU): reads gate+up [M,2·INTERMEDIATE], writes activated [M,INTERMEDIATE] → HBM
down GEMM (per expert M=64, N=HIDDEN=6144, K=INTERMEDIATE=2048, count=256):
    reads activated (A), projects to hidden
```
Fused `swiglu_down`: the down GEMM reads the gate+up tensor directly, applies SwiGLU on
chip to form the activated A tiles, then contracts over K=INTERMEDIATE. The activated
tensor is never materialized. This is a **prologue** fusion that widens down's A input to
`2·INTERMEDIATE` (gate+up) — modeled with `common.widened_a_points(a_width_mult=2)`.

## Fused-kernel model (`model.py`)

Per-expert dims: `M=64, N=HIDDEN=6144, K=INTERMEDIATE=2048`, count=256.

- `tensor_operations = 2·M·N·K` (= baseline down).
- `cuda_operations = M·INTERMEDIATE·SWIGLU_FLOPS_PER_ELEMENT` (= baseline activation per
  expert; ×count = 268.435 MFLOP). **FLOPs conserved** (down is per-dispatched-token, same
  as activation — no redundancy).
- Traffic via `widened_a_points(a_width_mult=2)`:
  - A tiles read gate+up (2·m0·k0), SwiGLU'd on chip
  - `final_tile_bytes = m0·n0·bpe` (raw down output; combine/residual stay separate)
  - `aux_hbm_bytes = 0`, `aux_buffer_bytes = 0`
- Removes baseline stages: `activation` (vector), `down` (GEMM).

### Expected traffic delta (hand check)
- Baseline: activation 192 MiB + down's activated A-reads `D_A`.
- Fused: down reads gate+up (≈ `2·D_A`), activation gone.
- Saved ≈ `192 − D_A`. down is weight-dominated (6400 MiB, W≈6144, output 192,
  `D_A ≈ 64 MiB` if A read ~once), so **saved ≈ 128 MiB** — less than Fusion 4's 256,
  because fusing the activation into down's prologue **doubles** down's input read (gate+up
  vs activated). If down re-reads A across many N-tiles, the doubling could erode or even
  outweigh the 192 MiB activation saving — the Snowcat frontier + best-C will reveal the
  net. This is the key contrast with Fusion 4 (epilogue fusion, strictly cheaper).

## Verification
- Clean run under `conda run -n fusion python -m fusion.activation_down.analysis`.
- FLOP conserved (down + activation).
- Compare saving to Fusion 4; confirm the prologue-widening cost via the fused down's
  A-read growth.
