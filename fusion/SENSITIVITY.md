# Hardware Sensitivity to Fusion — GPU performance sweep

**Question:** across GPU performance setups (HBM latency, HBM bandwidth, tensor-core
GFLOP/s, CUDA-core GFLOP/s), when does applying a kernel fusion actually change the
outcome — the **optimal die-area split** and/or the **total time** — and when is it
irrelevant? Everything is compared to the **default** setup
(2.04 TB/s, 500 cyc, 512 GFLOP/s tensor, 5.64 GFLOP/s CUDA).

Method: `fusion/sensitivity.py`. Each of the four knobs is swept one at a time (values from
the models' existing sensitivity tables) while the others stay at default; at every setting
we optimise the die split for the unfused baseline and for each of the six fusions, over
both **decode** (memory-bound) and **prefill** (compute/tensor-bound), and record:
- **Δsplit** = unfused-optimal CUDA cores − fused-optimal CUDA cores (die-partition shift);
- **Δtime%** = fusion's change in total layer time;
- ΔHBM (secondary; see the fusion `SUMMARY*.md`).

The Snowcat traffic frontiers are hardware-independent, so they're built once per stage and
reused across all settings. The area grid is coarsened to 0.002 for the sweep, which
quantises **Δsplit to ~55 CUDA cores** — read the split-shift numbers as trends, not to the
core. Metrics below are the **max over the six fusions** unless noted.

![sensitivity](result/sensitivity.png)

## Answer 1 — Die-area distribution: when does fusion change the optimal split?

Only the three **activation-removing FFN fusions (F4 up_gate+act, F5 act+down, F6 full FFN)**
ever move the split. The attention-boundary fusions (F1/F2/F3) *never* change it (Δsplit = 0
in every setup) — their compute hides under existing roofs.

**Decode (memory-bound) is the fusion-sensitive regime**, and the split shift is governed by:

| Knob | Behaviour of the split shift | Interpretation |
|---|---|---|
| **CUDA GFLOP/s (dominant)** | **inverse**: 2.82→**+327**, 4.23→+218, 5.64→+164, 8.46→+109, 11.28→+109 cores | Weak CUDA cores make the SwiGLU activation expensive in *area* (many weak cores); fusing it away frees the most → biggest partition change. |
| **HBM bandwidth** | **non-monotonic**: 1.02→+55, 1.53→+109, 2.04→+164, 3.06→**+218**, 4.08→**0** | Rises with bandwidth, then **collapses at 4.08 TB/s** — past there the decode FFN GEMMs stop being memory-bound, so fusion no longer changes the split. |
| **Tensor GFLOP/s** | **threshold**: 256→**0**, 384→+164, 512→+164, 768→+109, 1024→+109 | Below ~384 GFLOP/s the FFN GEMMs turn compute-bound and the split stops moving; adequate tensor throughput is required for sensitivity. |
| **HBM latency** | flat (no effect) | The `num_stages` model hides latency; fusion doesn't interact with it. |

**Prefill (compute/tensor-bound) is split-insensitive**: Δsplit = 0 everywhere except at the
weakest tensor throughput (256 GFLOP/s → +109), where prefill tips even more tensor-bound.
Prefill's CUDA-core count is set by the DSA lightning-indexer's gate/top-k, not the FFN, so
removing the activation doesn't change it.

## Answer 2 — Total time: when does fusion change runtime?

| | Decode | Prefill |
|---|---|---|
| Typical Δtime (best fusion) | **~0.017%** (fusion-insensitive) | **~0.5–0.77%** |
| vs bandwidth | flat; **F6 flips to −0.18% at 4.08 TB/s** (fusion *hurts*) | **grows at low bandwidth** (1.02→0.77% vs 2.04→0.57%) |
| vs tensor GFLOP/s | flat | **grows at high tensor** (256→0.49% vs 1024→0.76%) |
| vs CUDA GFLOP/s | flat | flat |
| vs latency | flat | flat |

- **Prefill is ~30× more time-sensitive to fusion than decode** — the removed activation
  kernel is a real fraction of prefill, whereas decode is ~99% attention-core (KV-cache)
  bound, so fusion barely moves its clock.
- Prefill's time win **grows where the removed vector/activation kernels become a bigger
  relative cost**: low HBM bandwidth (their memory time matters more) and high tensor
  throughput (the GEMMs finish faster, so the leftover vector work dominates).
- **Anti-case:** at 2× bandwidth (4.08 TB/s) the **full-FFN fusion F6 makes decode 0.18%
  *slower*** — decode is no longer memory-bound enough to hide F6's weight-reread traffic,
  so the fusion becomes counterproductive. (Consistent with the prefill finding that F6 is a
  large-M / high-throughput anti-pattern.)
- **HBM latency changes nothing** for either stage.

## Bottom line — the fusion-sensitivity map

- **Most sensitive setup (area distribution):** **memory-bound decode with weak CUDA cores.**
  There, fusing the SwiGLU activation swings the optimal CUDA-core count by hundreds — the
  die *should* be partitioned differently depending on whether you fuse. Sensitivity also
  needs **moderate–high bandwidth and adequate (≥384 GFLOP/s) tensor throughput**; it
  vanishes at very high bandwidth (≥4 TB/s) or weak tensor cores.
- **Most sensitive setup (time):** **compute-bound prefill at low bandwidth and/or high
  tensor throughput** — but still < 1%.
- **Least sensitive setups:** compute-bound **prefill** for the split (never moves, bar
  extreme weak-tensor), and **very-high-bandwidth decode** for both (fusion stops helping,
  and F6 starts hurting).
- **Irrelevant knob:** **HBM latency** — no effect on fusion's value in any regime.

**Design implication:** fusion is a *partitioning-relevant* decision only when the workload
is memory-bound (decode) **and** CUDA cores are the scarce/weak resource — exactly where
removing the CUDA-side SwiGLU frees the most silicon. In compute-bound (prefill) or
bandwidth-rich regimes, fusion is at best a sub-1% time tweak and never changes how the die
should be split (and the full-FFN fusion can even backfire).

## Reproduce
```
conda run -n fusion python -m fusion.sensitivity            # writes result/sensitivity.json
conda run -n fusion python -m fusion.make_sensitivity_figure
```
