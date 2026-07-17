# Prefill Fusion Area-Distribution Analysis — Summary (all six fusions)

The same six fused kernels as the decode study (`SUMMARY.md`), re-evaluated over the full
GLM-5.2 **prefill** layer (`prefill_area_latency.py`: MLA + DeepSeek Sparse Attention,
K/V materialized, **compute/tensor-bound**; single 1,048,576-token prompt, so every GEMM's
M = tokens, up_gate/down M = 32,768/expert). Only one kernel is fused per analysis; all
compared to the fully-unfused prefill baseline. Method: `notes/general_fusion_analysis.md`;
per-fusion detail: each `fusion/<name>/report_prefill.md`.

HBM is reported under the **min-traffic-among-time-optimal** convention (a sane scheduler
minimises traffic when it is free). This is necessary because prefill kernels are
compute-bound — many tilings tie on time, so the raw frontier traffic is otherwise
ill-defined. (For memory-bound decode the convention is a no-op; decode's headline
numbers are unchanged except the two compute-bound `mla_o` fusions F1/F2 — see `SUMMARY.md`.)

**Unfused baseline optimum:** rc 0.012 / rt 0.945 / SMEM 8.086 MiB → **326 CUDA / 858
tensor cores**, total **12992.4 ms**, 1729.7 GiB HBM (min-traffic), 434.5 TFLOP/s
(≈ tensor roof).

## Results

| # | Fusion | Time saved | Layer HBM change | Optimal CUDA/Tensor | FLOPs |
|---|--------|---:|---:|:--:|:--:|
| — | unfused baseline | — | — | 326 / 858 | — |
| 1 | FlashAttention + residual | 18.95 ms (0.146%) | **+72 GiB** (increase) | 326 / 858 | conserved |
| 2 | FlashAttention + residual + RMSNorm | 25.96 ms (0.200%) | **+60 GiB** (increase) | 326 / 858 | conserved |
| 3 | RMSNorm + up_gate | 7.01 ms (0.054%) | −12 GiB (saved) | 326 / 858 | +0.02% redundant |
| 4 | up_gate + activation | 74.75 ms (0.575%) | **−128 GiB** (saved) | 326 / 858 | conserved |
| 5 | activation + down | 74.75 ms (0.575%) | +32 GiB (increase) | 326 / 858 | conserved |
| 6 | up_gate + activation + down | 74.75 ms (0.575%) | **+384 GiB** (increase) | 326 / 858 | conserved |

![summary](result/fusion_summary_prefill.png)

## Findings

1. **No fusion moves the die partition.** All six keep the prefill optimum at 326 CUDA /
   858 tensor. Prefill is tensor-bound: the fusions don't change the tensor GEMM workload,
   and the one CUDA-side kernel they remove (SwiGLU activation) doesn't free enough CUDA
   demand to matter — the CUDA count is set by the DSA lightning-indexer's gate/top-k, not
   the FFN activation. **This is the sharpest decode/prefill contrast**: in decode, F4/F6
   shifted the split toward tensor cores (490→381→354 CUDA); in prefill the split is
   invariant.

2. **Time wins are small and are just "remove the vector/CUDA kernel."** All GEMMs stay
   tensor-bound, so fusing them changes neither their tensor time nor the split; the only
   time saved is the removed activation / residual / rmsnorm kernels (0.05–0.58% of the
   layer, which is dominated ~81% by the DSA attention core).

3. **HBM is hidden, and often *increases*.** Because prefill is compute-bound, HBM never
   affects time — so the fusions freely trade it. Two fusions **save** HBM (F4 −128 GiB
   writing activated-only; F3 −12 GiB removing the rmsnorm read — both add no GEMM SMEM
   pressure). Four **increase** HBM, all hidden:
   - F1/F2: the residual/RMS epilogue's on-chip output-tile buffer starves the huge
     compute-bound `mla_o` GEMM → worse tiling (+72 / +60 GiB).
   - F5: down's prologue reads the 2×-wide gate+up (+32 GiB).
   - **F6: +384 GiB — the weight-reread catastrophe.** At M=32,768 the fused FFN's row-block
     can't hold enough rows, so both weight matrices are re-read ~64×.

4. **Epilogue beats prologue, again.** Same SwiGLU: **F4 (into up_gate epilogue) −128 GiB**
   vs **F5 (into down prologue) +32 GiB** — a 160 GiB swing. Fuse SwiGLU into up_gate.

5. **Full FFN fusion (F6) is decode-only.** Decode (M=64): the strongest fusion (−384 MiB).
   Prefill (M=32,768): counterproductive (+384 GiB weight re-reads). This is the
   quantitative reason `ffn_fused_area_latency.py` fuses only up_gate+SwiGLU and leaves
   down standard for large-M regimes.

## Decode vs prefill (same six fusions)

| | Decode (batch 2048, memory-bound) | Prefill (1M prompt, tensor-bound) |
|---|---|---|
| Does any fusion move the split? | **Yes** (F4/F5→381, F6→354 CUDA) | **No** (all 326/858) |
| HBM savings realized in time? | Yes (memory-bound) | No (hidden under compute) |
| Best FFN fusion | F6 full-FFN (−384 MiB) | F4 up_gate+SwiGLU (−128 GiB); **F6 is harmful** |
| F1/F2 (mla_o + residual/RMS) | negligible; slight HBM ↑ (starvation) | slight time win; HBM ↑ (starvation), hidden |
| Governing lever | HBM bandwidth | tensor-core throughput |

**Bottom line for prefill:** fusion is a modest *time* optimisation (remove the small
vector/activation kernels) with **no effect on how the die should be partitioned** — the
tensor-heavy prefill optimum (858 tensor / 326 CUDA, ~8 MiB SMEM) is invariant to all six
fusions. Only F4 (up_gate+SwiGLU) and F3 (RMSNorm+up_gate) also reduce HBM; the residual-,
down-prologue-, and full-FFN fusions *increase* it (harmlessly, since compute-bound), with
the full-FFN fusion being actively wasteful at prefill's large M.

## Reproduce

```
conda run -n fusion python -m fusion.<name>.analysis_prefill   # same six <name>s as decode
conda run -n fusion python -m fusion.make_summary_figure_prefill
```
Outputs go to `fusion/<name>/result/prefill_*` (gitignored).
