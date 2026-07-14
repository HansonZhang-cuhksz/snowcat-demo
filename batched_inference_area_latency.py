"""Batched end-to-end single-layer inference area analysis.

For an ``N``-batched serving load (N concurrent sessions):
  - **prefill** each prompt -> run single-sequence prefill ``N`` times, and
  - **decode** all N sessions together -> N-batched decode for ``DECODE_TOKENS`` steps.

    inference_time(area, N) = N * prefill_time(area)                       # N prefills
                            + sum_{t} decode_step_time(area, batch=N, S+t)  # batched decode

Prefill (``prefill_area_latency``, 1 prompt) is compute/tensor bound, so N prefills cost
N x a single prefill.  Decode (``decode_area_latency``, batch N) is HBM-bandwidth bound:
each step reads N KV caches (attention traffic ~ N) but amortizes the MoE expert weights
across the N tokens.  Both stage models share the chip constants and area grid, so their
per-node arrays line up; decode attention is linear in context, so the growing steps sum
analytically.

The question: does batching shift the single best area split away from the (tensor-heavy)
prefill optimum?
"""

from __future__ import annotations

import numpy as np

import decode_area_latency as dc
import prefill_area_latency as pf

# --- Batched serving scenario (tune these) ---
BATCH_N = 1                    # concurrent sessions (prefills, and decode batch)
PROMPT_SEQ_LEN = 1_048_576     # prompt length per session
DECODE_TOKENS = 150            # output tokens generated per session

# True (default) runs both stage models with the Snowcat/Orojenesis traffic
# frontier.  False propagates no-snowcat mode to both stages: algorithmic-minimum
# GEMM traffic (OI independent of SMEM) and BW_eff = min(bw, SMEM/latency).
USE_SNOWCAT = True


def _fmt_split(res: dict, i: int) -> str:
    return (
        f"rc={res['rc'][i]:.3f} rt={res['rt'][i]:.3f} r_smem={res['r_smem'][i]:.3f} | "
        f"SMEM {res['smem_bytes'][i] / 2**20:.3f} MiB | "
        f"{int(res['cuda_cores'][i])} CUDA + {int(res['tensor_cores'][i])} tensor cores"
    )


def evaluate_batched(batch_n: int, decode_tokens: int, seq_len: int) -> dict:
    """Evaluate N prefills + N-batched decode over the shared area grid."""
    pf.USE_SNOWCAT = USE_SNOWCAT
    dc.USE_SNOWCAT = USE_SNOWCAT
    pf.configure(1, seq_len)            # single-sequence prefill (run N times)
    dc.configure(batch_n, seq_len)      # N-batched decode

    prefill = pf.evaluate_layer()
    decode = dc.evaluate_layer()
    if not np.array_equal(prefill["rc"], decode["rc"]):
        raise RuntimeError("prefill and decode area grids are not aligned")

    prefill_time = batch_n * prefill["total_time"]          # N prefills (compute-bound)
    decode_step = decode["total_time"]                       # one batched decode step at S
    decode_attn = decode["attention_time"]                   # linear in context

    n = decode_tokens
    s = seq_len
    context_growth = n + n * (n - 1) / (2.0 * s)
    with np.errstate(invalid="ignore"):
        decode_total = n * (decode_step - decode_attn) + decode_attn * context_growth
        combined = prefill_time + decode_total
    best = int(np.nanargmin(combined))

    combined_ops = batch_n * prefill["modeled_operations"] + n * decode["modeled_operations"]
    # tokens processed = N prompts prefilled + N*decode_tokens generated
    tokens = batch_n * seq_len + batch_n * n

    return {
        "prefill": prefill, "decode": decode, "batch_n": batch_n,
        "prefill_time": prefill_time, "decode_total": decode_total,
        "combined": combined, "best": best, "combined_ops": combined_ops, "tokens": tokens,
    }


def _report(res: dict) -> None:
    p = res["prefill"]
    i = res["best"]
    n = res["batch_n"]
    pf_t = float(res["prefill_time"][i])
    dc_t = float(res["decode_total"][i])
    total = float(res["combined"][i])
    print(f"\n=== Batched inference: N={n} sessions, {PROMPT_SEQ_LEN:,}-token prompts, "
          f"{DECODE_TOKENS} output tokens ===")
    if not USE_SNOWCAT:
        print("Traffic model: NO SNOWCAT -- algorithmic-minimum HBM traffic "
              "(OI independent of SMEM; BW_eff = min(bw, SMEM/latency))")
    print(f"Combined-optimal split: {_fmt_split(p, i)}")
    print(f"  {n} prefills : {pf_t * 1e3:12.1f} ms  ({100 * pf_t / total:5.1f}%)")
    print(f"  decode x{DECODE_TOKENS} (batch {n}): {dc_t * 1e3:12.1f} ms  ({100 * dc_t / total:5.1f}%)")
    print(f"  TOTAL      : {total * 1e3:12.1f} ms  ({total:.3f} s)")
    print(f"  throughput : {res['combined_ops'] / total / 1e12:.1f} TFLOP/s, "
          f"{res['tokens'] / total / 1e3:.1f} K tokens/s")


def sweep(batches: list[int]) -> None:
    print(f"Prompt {PROMPT_SEQ_LEN:,} tokens, {DECODE_TOKENS} output tokens/session; "
          f"prefill DSA, decode dense."
          + ("" if USE_SNOWCAT else "  [NO SNOWCAT: algorithmic-min traffic]") + "\n")
    hdr = (f"{'N':>5} | {'total ms':>11} | {'prefill%':>8} {'decode%':>8} | {'rt':>5} "
           f"{'CUDA':>4} {'TENS':>5} {'SMEM MiB':>8} | {'Ktok/s':>8} | {'ms/session':>10}")
    print(hdr)
    print("-" * len(hdr))
    for n in batches:
        r = evaluate_batched(n, DECODE_TOKENS, PROMPT_SEQ_LEN)
        p = r["prefill"]
        i = r["best"]
        total = float(r["combined"][i])
        pf_t = float(r["prefill_time"][i])
        dc_t = float(r["decode_total"][i])
        print(f"{n:>5} | {total * 1e3:>11.1f} | {100 * pf_t / total:>7.1f}% {100 * dc_t / total:>7.1f}% | "
              f"{p['rt'][i]:>5.3f} {int(p['cuda_cores'][i]):>4} {int(p['tensor_cores'][i]):>5} "
              f"{p['smem_bytes'][i] / 2**20:>8.2f} | {r['tokens'] / total / 1e3:>8.1f} | "
              f"{total / n * 1e3:>10.1f}")


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="GLM-5.2 batched single-layer inference (N prefills + N-batched "
        "decode) area analysis.")
    parser.add_argument("--batch", type=int, default=BATCH_N, help="concurrent sessions.")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--seq-len", type=int, default=PROMPT_SEQ_LEN)
    parser.add_argument("--sweep", type=lambda s: [int(x) for x in s.split(",")],
                        default=None, help="comma-separated batch sizes to sweep.")
    parser.add_argument(
        "--no-snowcat", action="store_true",
        help="disable the Snowcat traffic frontier in both stage models "
        "(algorithmic-minimum GEMM traffic; overly optimistic).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    BATCH_N = args.batch
    DECODE_TOKENS = args.decode_tokens
    PROMPT_SEQ_LEN = args.seq_len
    if args.no_snowcat:
        USE_SNOWCAT = False
    if args.sweep:
        sweep(args.sweep)
    else:
        _report(evaluate_batched(BATCH_N, DECODE_TOKENS, PROMPT_SEQ_LEN))
