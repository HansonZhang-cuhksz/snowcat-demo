"""End-to-end single-layer inference area analysis: prefill once + decode N tokens.

Concatenates the two stage models on ONE fixed die-area budget (the chip is designed
once, then runs both stages):

    inference_time(area) = prefill_time(area) + sum_{t=0..N-1} decode_step_time(area, S+t)

- Prefill: ``prefill_area_latency`` -- one prompt of ``PROMPT_SEQ_LEN`` tokens (GLM-5.2
  MLA + DeepSeek Sparse Attention, compute/tensor bound).
- Decode: ``decode_area_latency`` -- ``DECODE_TOKENS`` autoregressive steps at batch 1,
  each attending to a KV cache that grows from S to S+DECODE_TOKENS-1 (absorbed MLA,
  bandwidth bound).  (This decode model is dense over the KV cache; DSA-decode would be a
  refinement.)

Both stage models share identical chip constants and area grid, so their per-node arrays
line up element-wise.  Decode's attention time is exactly linear in context length (both
its compute and memory terms scale with the number of KV positions), so the N growing
steps are summed analytically from a single decode evaluation -- no need to re-run the
grid per token.

The single best area split for the *combined* workload is reported, along with each
stage's standalone optimum, to expose the prefill(tensor)-vs-decode(bandwidth/SMEM)
tension.
"""

from __future__ import annotations

import numpy as np

import decode_area_latency as dc
import prefill_area_latency as pf

# --- Inference scenario (tune these) ---
PROMPT_SEQ_LEN = 1_048_576      # prefill prompt length (GLM-5.2 max context)
DECODE_TOKENS = 150            # number of tokens generated autoregressively
DECODE_BATCH = 1              # decode batch = concurrent sequences


def _fmt_split(res: dict, i: int) -> str:
    return (
        f"rc={res['rc'][i]:.3f} rt={res['rt'][i]:.3f} r_smem={res['r_smem'][i]:.3f} | "
        f"SMEM {res['smem_bytes'][i] / 2**20:.3f} MiB | "
        f"{int(res['cuda_cores'][i])} CUDA + {int(res['tensor_cores'][i])} tensor cores"
    )


def evaluate_inference() -> dict:
    """Evaluate prefill + N-token decode over the shared area grid.  Returns per-node
    arrays and the combined-best index."""
    pf.configure(1, PROMPT_SEQ_LEN)
    dc.configure(DECODE_BATCH, PROMPT_SEQ_LEN)

    prefill = pf.evaluate_layer()
    decode = dc.evaluate_layer()

    if not np.array_equal(prefill["rc"], decode["rc"]):
        raise RuntimeError("prefill and decode area grids are not aligned")

    prefill_time = prefill["total_time"]                    # [grid]
    decode_step = decode["total_time"]                       # [grid] at context = S
    decode_attn = decode["attention_time"]                   # [grid], linear in context

    # Sum over N steps with context S, S+1, ..., S+N-1.  attention(c) = attention(S)*c/S.
    n = DECODE_TOKENS
    s = PROMPT_SEQ_LEN
    context_growth = n + n * (n - 1) / (2.0 * s)             # sum(c_t)/S = N + N(N-1)/2S
    # inf - inf = nan at degenerate area corners (0 cores); those nodes are excluded by
    # nanargmin below, so ignore the resulting invalid-op warnings.
    with np.errstate(invalid="ignore"):
        decode_base = decode_step - decode_attn              # context-independent per step
        decode_total = n * decode_base + decode_attn * context_growth
        combined = prefill_time + decode_total
    best = int(np.nanargmin(combined))

    # per-step decode ops are ~context-independent (GEMMs dominate ops); approximate
    # combined ops for a throughput number.
    combined_ops = prefill["modeled_operations"] + n * decode["modeled_operations"]

    return {
        "prefill": prefill,
        "decode": decode,
        "prefill_time": prefill_time,
        "decode_total": decode_total,
        "decode_step": decode_step,
        "combined": combined,
        "best": best,
        "combined_ops": combined_ops,
        "context_growth": context_growth,
    }


def plot_combined(res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from pathlib import Path

    Path("result").mkdir(exist_ok=True)
    p = res["prefill"]
    combined = res["combined"]
    valid = np.isfinite(combined) & (combined > 0)
    plt.figure(figsize=(10, 7))
    c_ms = combined[valid] * 1e3
    sc = plt.scatter(p["rt"][valid], p["rc"][valid], c=c_ms, s=8, cmap="viridis_r",
                     norm=LogNorm(vmin=c_ms.min(), vmax=c_ms.max()))
    plt.colorbar(sc, label="Combined inference time (ms)")
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title(f"GLM-5.2 Inference (prefill + {DECODE_TOKENS} decode) Time vs Area Split")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main(write_outputs: bool = True) -> dict:
    res = evaluate_inference()
    p, d = res["prefill"], res["decode"]
    i = res["best"]
    pi, di = p["best_index"], d["best_index"]

    prefill_at_best = float(res["prefill_time"][i])
    decode_at_best = float(res["decode_total"][i])
    combined_best = float(res["combined"][i])

    print("\n=== Inference Scenario ===")
    print(f"Prompt length (prefill): {PROMPT_SEQ_LEN:,} tokens")
    print(f"Decode tokens: {DECODE_TOKENS} (batch {DECODE_BATCH}); "
          f"KV context grows {PROMPT_SEQ_LEN:,} -> {PROMPT_SEQ_LEN + DECODE_TOKENS - 1:,}")
    print(f"Prefill attention: {'DSA top-' + str(pf.DSA_INDEX_TOPK) if pf.DSA_ENABLED else 'dense'}; "
          f"decode attention: dense absorbed MLA over the KV cache")

    print("\n=== Combined-Optimal Area Split (single chip for both stages) ===")
    print(f"  {_fmt_split(p, i)}")
    print(f"  prefill  : {prefill_at_best * 1e3:12.3f} ms  ({100 * prefill_at_best / combined_best:5.1f}%)")
    print(f"  decode x{DECODE_TOKENS}: {decode_at_best * 1e3:12.3f} ms  ({100 * decode_at_best / combined_best:5.1f}%)")
    print(f"  per decode token: {decode_at_best / DECODE_TOKENS * 1e3:.4f} ms")
    print(f"  TOTAL    : {combined_best * 1e3:12.3f} ms  ({combined_best:.4f} s)")
    print(f"  combined throughput: {res['combined_ops'] / combined_best / 1e12:.3f} TFLOP/s")

    print("\n=== Standalone stage optima (for comparison) ===")
    print(f"  prefill-alone best : {float(p['total_time'][pi]) * 1e3:12.3f} ms  @ {_fmt_split(p, pi)}")
    print(f"  decode-alone  best : {float(d['total_time'][di]) * 1e3:12.4f} ms/token @ {_fmt_split(d, di)}")

    print("\n=== Which stage governs the shared design? ===")
    # combined time if the chip were tuned for prefill-only vs decode-only splits
    for label, idx in (("prefill-optimal split", pi), ("decode-optimal split", di),
                        ("combined-optimal split", i)):
        c = float(res["combined"][idx])
        print(f"  {label:<24}: combined {c * 1e3:11.3f} ms "
              f"(prefill {float(res['prefill_time'][idx]) * 1e3:.1f} + "
              f"decode {float(res['decode_total'][idx]) * 1e3:.1f}); "
              f"{100 * (c - combined_best) / combined_best:+.2f}% vs best")

    if write_outputs:
        plot_combined(res, "./result/inference_area_latency_combined_time.png")
        print("\nWrote ./result/inference_area_latency_combined_time.png")

    return res


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="GLM-5.2 single-layer end-to-end inference (prefill + N-token decode) "
        "area analysis on one fixed die budget."
    )
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS,
                        help=f"tokens to decode (default {DECODE_TOKENS}).")
    parser.add_argument("--seq-len", type=int, default=PROMPT_SEQ_LEN,
                        help=f"prompt length for prefill (default {PROMPT_SEQ_LEN}).")
    parser.add_argument("--no-write", action="store_true", help="skip the PNG output.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    DECODE_TOKENS = args.decode_tokens
    PROMPT_SEQ_LEN = args.seq_len
    main(write_outputs=not args.no_write)
