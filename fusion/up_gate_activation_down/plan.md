# Plan — Fusion 6: up_gate + activation + down (full FFN GEMM-GEMM fusion)

See `notes/general_fusion_analysis.md` for shared method/decisions.

## What is fused

The entire expert FFN as one kernel producing `out[M,HIDDEN]` from `x[M,HIDDEN]`:
```
up_gate: x @ W_ug        → gate+up[M, 2·INTERMEDIATE]
SwiGLU:                  → activated[M, INTERMEDIATE]
down:    activated @ W_dn → out[M, HIDDEN]
```
Fused `ffn_up_gate_swiglu_down`: the gate+up and activated intermediates **never touch
HBM** (CODA on-chip intermediate). Note this is a **GEMM-GEMM** fusion, unlike
`ffn_fused_area_latency.py` which fuses only up_gate+SwiGLU and keeps down standard.

## The row-block tradeoff (why this fusion is SMEM-gated)

down contracts over the full INTERMEDIATE dim, so producing any `out` row needs the full
`activated[m0, :]` row resident. The kernel processes a row-block of `m0` tokens; for each
block it reads **both** weight matrices once (`W_ug` + `W_dn`). With `mt = M/m0` blocks,
weight traffic = `mt·(W_ug + W_dn)`. So:
- large `m0` (few blocks) → weights read ~once, but a big on-chip resident
  (`activated + out` accumulators ≈ `m0·(INTERMEDIATE+HIDDEN)` elements);
- small `m0` → little SMEM, but weights re-read `mt×` → traffic explodes.

The fusion only helps when SMEM holds a large enough `m0` to avoid weight re-reads — this
is exactly why the repo's `ffn_fused` leaves down un-fused. Modeled by enumerating `m0`.

## Fused-kernel model (`model.py`, custom)

Per expert `M=64`, `HIDDEN=6144`, `INTERMEDIATE=2048`, count=256. Enumerate
`m0 ∈ divisors(M)`, `m0 ≥ TENSOR_CORE_MIN_BM` (16 → {16,32,64}); `mt = M/m0`:

- `traffic(m0) = 2·M·HIDDEN·bpe` (x read + out write) `+ mt·(W_ug + W_dn)`,
  with `W_ug = HIDDEN·2·INTERMEDIATE·bpe`, `W_dn = INTERMEDIATE·HIDDEN·bpe`. Intermediate = 0.
- `buffer(m0) = m0·(INTERMEDIATE+HIDDEN)·bpe` (peak resident: activated + out accumulators)
  `+ 2·INTERMEDIATE·bpe + HIDDEN·bpe` (W_ug/W_dn K-slice tiles) `+ HIDDEN·bpe` (x row).
- `tensor_operations = 2·M·(2·INTERMEDIATE)·HIDDEN + 2·M·HIDDEN·INTERMEDIATE` (up_gate + down).
- `cuda_operations = M·INTERMEDIATE·SWIGLU_FLOPS_PER_ELEMENT` (SwiGLU). **FLOPs conserved.**
- Removes baseline stages: `up_gate`, `activation`, `down`.

Assumption (documented): register-accumulator ideal — weights + x read once per `m0`
block, full accumulators resident. Finer sub-tilings (trading buffer for x/weight
re-reads) would add intermediate frontier points; the `m0` knob captures the dominant
weight-reread-vs-SMEM tradeoff that governs whether the fusion helps.

### Expected delta
If SMEM holds `m0=64` (buffer ≈ 1.0 MiB, mt=1): fused traffic = 18816 MiB vs baseline
up_gate+activation+down 19200 MiB → **saved 384 MiB** (the entire gate+up + activated
round-trip — the max of all six fusions). Needs ~1 MiB SMEM; below that, `m0<64` forces
weight re-reads and the fusion loses badly.

## Verification
- Clean run under `conda run -n fusion python -m fusion.up_gate_activation_down.analysis`.
- FLOP conserved (up_gate + activation + down).
- Confirm the winning `m0` at the optimum and that traffic = 18816 MiB (saved 384);
  inspect what happens to the fused kernel if forced to small SMEM (weight-reread blowup).
