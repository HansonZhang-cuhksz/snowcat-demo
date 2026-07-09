# Small-batch / tensor-core-tile support for the area analyzers

Adds handling of small MoE batches so the analyzers no longer raise
"no tensor-core-compatible mapping" when an expert receives < 16 tokens.

Applied to all five area-distribution programs: `decode_area_latency.py`, `ffn_area.py`,
`ffn_area_latency.py`, `ffn_fused_area.py`, `ffn_fused_area_latency.py`. (Not `multi_gemm_area.py`,
out of scope.)

## Shared resolver: `expert_workload.py`

- `even_expert_token_split(batch, experts, top_k) -> EvenExpertSplit`: even routing with
  `total = batch*top_k`, `base, rem = divmod(total, experts)`. `rem` experts get `base+1` (ceil),
  `experts-rem` get `base` (floor). If `base == 0` (batch too small for every expert to get a token),
  only `rem` experts are active with 1 token each; the rest are idle (dropped). `.summary()` prints it.
- `padded_m(tokens, min_bm=16)`: `tokens` if `>=16` else `16`.
- `padded_gemm_groups(split, 16) -> [(M_padded, expert_count)]` merged by M (for GEMMs whose cost
  depends only on M).

## Decisions (from the user)

1. **M=16 padding is GEMM-tiling-only.** The per-expert up_gate/down GEMM ops+traffic use `M=max(t,16)`;
   the activation/SwiGLU and expert-combine stages use the **real** token counts. In the unfused files
   the SwiGLU is a separate `ACTIVATION_TASK` now sized by `total_assignments = batch*top_k`
   (elements, count=1). In the fused files the up_gate stage's `tensor_operations`/traffic use padded M
   while `cuda_operations` (SwiGLU epilogue) use the real token count.
2. **Unify on `ROUTER_TOP_K` as the input** (all 5 files). `TOKENS_PER_EXPERT` is now derived
   (`= EVEN_SPLIT.floor_tokens`, kept only as a reference/print). Enables the reduced-expert case.
3. **Uneven (`batch*top_k` not a multiple of experts):** ceil/floor split, printed at output.
   E.g. batch 100 -> `224 experts x 3 tokens, 32 experts x 4 tokens`. Fused builds one up_gate stage
   per token group (`up_gate_rms_swiglu_m3_x224`, `_m4_x32`); unfused merges GEMMs by padded M.

Also: any GEMM whose M is the batch (router, and the MLA projection/absorb GEMMs in the decode file)
is padded to 16 when `batch < 16`. The random-expert-distribution path pads M too (fixes its previous
crash on the <16-token binomial tail).

## Reporting

Each file prints `Expert token split: <summary>` and, when tokens/expert < 16, a note that per-expert
GEMM M is padded to 16 (tensor core underutilized). GEMM stage labels carry the padded M / active
count, e.g. `up_gate_x256`, or `up_gate_m16_x128` in multi-group cases.

## Verification (conda env `profiling`; numpy+matplotlib installed there)

Default regressions reproduce exactly:
| file | time | TFLOP/s |
|---|---|---|
| ffn_area.py / ffn_area_latency.py | 10.613 ms | 234.406 |
| ffn_fused_area.py / ffn_fused_area_latency.py | 10.350 ms | 240.382 |
| decode_area_latency.py (batch 2048, seq 1,048,576) | 1224.492 ms | 246.367 |

Small batches now run end-to-end (no error): e.g. decode batch 4/1; ffn_area batch 32 (padded, 9.87 ms);
ffn_fused_area_latency batch 16 (128 active experts, 5.09 ms); uneven batch 100 (ceil/floor split).
Random-expert path no longer crashes at batch 1024.

## STATUS: DONE (all 5 files + shared expert_workload.py)
