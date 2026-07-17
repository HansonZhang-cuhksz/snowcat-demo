# Plan — Fusion 4: up_gate + activation (SwiGLU)

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

```
up_gate GEMM (per expert M=64, N=2·INTERMEDIATE=4096, K=6144, count=256):
    writes raw gate+up [M, 4096] → HBM
activation (SwiGLU): reads gate+up, writes SiLU(gate)*up [M, INTERMEDIATE=2048] → HBM
```
Fused `up_gate_swiglu`: up_gate produces gate+up tiles on chip and applies SwiGLU in the
epilogue, writing only the activated [M, INTERMEDIATE] output. The raw gate+up is never
written / re-read. CODA `_traffic_for_pairwise_output_tile` (GEMM produces `2·p`
interleaved gate/up cols, epilogue stores `p` activated cols).

## Fused-kernel model (`model.py`)

Per-expert dims: `M=64, p=INTERMEDIATE=2048` (so `N=2p=4096`), `K=HIDDEN=6144`, count=256.

- `tensor_operations = 2·M·(2p)·K` (= baseline up_gate).
- `cuda_operations = M·p·SWIGLU_FLOPS_PER_ELEMENT` (= baseline activation, per expert;
  ×count = 268.435 MFLOP total). **FLOPs conserved** (no dispatch redundancy — activation
  is already per-dispatched-token).
- Traffic via `pairwise_tile_points`:
  - `final_tile_bytes = m0·p0·bpe`   (write activated output only)
  - `aux_hbm_bytes = 0`, `aux_buffer_bytes = 0` (SwiGLU is elementwise on the on-chip tile)
- Removes baseline stages: `up_gate` (GEMM), `activation` (vector).

### Expected traffic delta (hand check, per-expert ×256; bpe=2)
Round-trips on gate/up/activated (INTERMEDIATE=2048, batch·top_k rows = 16384):
- Baseline: up_gate writes gate+up (128 MiB) + activation reads gate (64) + reads up (64)
  + writes activated (64) = **320 MiB**.
- Fused: writes activated (64 MiB) = **64 MiB**.
- **Saved ≈ 256 MiB** (= up_gate raw-output write 64 saved + full activation kernel 192).
  Larger than Fusions 1–3. Still small vs 2.3 TiB → layer optimum likely unchanged, but
  the FFN-side stages shrink more.

## Verification
- Clean run under `conda run -n fusion python -m fusion.up_gate_activation.analysis`.
- FLOP accounting conserved (up_gate + activation).
- Fused up_gate GEMM A/W reads == baseline up_gate (only the output write changes:
  activated p cols vs raw 2p cols); layer saving ≈ 256 MiB.
