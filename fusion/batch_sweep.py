"""Decode batch-size sweep: how does batch size change fusion optimality?

At the DEFAULT GPU spec (2.04 TB/s, 500 cyc, 512 / 5.64 GFLOP/s) and default KV context
(1,048,576), sweep the decode batch size and, at each batch, optimise the die-area split
for the unfused baseline and each of the six fusions, recording:

  * dtime%  -- fusion's change in total layer time
  * dcuda   -- die-partition shift (unfused optimal CUDA cores - fused optimal CUDA cores)
  * dhbm    -- HBM change (GiB, min-traffic convention)
  * plus per batch: tokens/expert and the attention/FFN time balance (context).

Batch changes TOKENS_PER_EXPERT = batch*top_k/EXPERTS and the GEMM dims, so the Snowcat
frontiers are rebuilt for every batch (they can't be cached across batches, unlike the
hardware sweep). We restrict to batch >= 32, where all 256 experts receive >=1 token so the
fusion models' count = EXPERTS holds; below that is the reduced-active-expert regime.

Run: conda run -n fusion python -m fusion.batch_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import decode_area_latency as dec

from fusion import common
from fusion.flash_attention_residual import model as f1
from fusion.flash_attention_residual_rmsnorm import model as f2
from fusion.rmsnorm_up_gate import model as f3
from fusion.up_gate_activation import model as f4
from fusion.activation_down import model as f5
from fusion.up_gate_activation_down import model as f6

FUSIONS = [
    ("F1 FA+resid", f1),
    ("F2 FA+resid+RMS", f2),
    ("F3 RMS+up_gate", f3),
    ("F4 up_gate+act", f4),
    ("F5 act+down", f5),
    ("F6 up_gate+act+down", f6),
]

BATCHES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_BATCH = 2048
SEQ_LEN = dec.DEFAULT_SEQ_LEN

DEF_BW, DEF_LAT, DEF_TF, DEF_CF = 2.04e12, 500, 512e9, 5.64e9
GRID = 0.002  # ~55-core split-shift quantization (trend-level; see SENSITIVITY.md)


def set_default_hw() -> None:
    dec.bw = DEF_BW
    dec.HBM_LATENCY_CYCLES = DEF_LAT
    dec.TENSOR_FLOPS = DEF_TF
    dec.ACTIVATION_FLOPS_PER_CUDA_CORE = DEF_CF
    common.bw = DEF_BW
    common.HBM_LATENCY_CYCLES = DEF_LAT
    common.LATENCY_SECONDS = DEF_LAT / dec.CUDA_CLOCK_HZ


def eval_batch(batch: int) -> list[dict]:
    dec.configure(batch, SEQ_LEN)
    dec.AREA_GRID_STEP = GRID
    set_default_hw()
    base = dec.evaluate_layer()
    bi = base["best_index"]
    cuda, tensor, smem = base["cuda_cores"], base["tensor_cores"], base["smem_bytes"]
    tensor_roof, cuda_roof = base["tensor_roof"], base["cuda_roof"]
    total_time = base["total_time"]
    unfused_time = float(total_time[bi])
    attn_frac = float(base["attention_time"][bi] / total_time[bi])

    # Baseline HBM (min-traffic), computed once and reused across fusions.
    stage_times = common.baseline_stage_times(base)
    vec_traffic = common._vector_stage_traffic(dec)
    agg, gw = base["aggregate_names"], base["group_weights"]
    gemm_min: dict[str, np.ndarray] = {}
    for fr in base["frontiers"]:
        a = agg[fr.label]
        gemm_min[a] = gemm_min.get(a, 0.0) + gw[fr.label] * common.baseline_gemm_min_traffic(
            fr, smem, tensor_roof
        )
    total_hbm_baseline = sum(gemm_min.values()) + sum(float(v) for v in vec_traffic.values())

    rows = []
    for label, model in FUSIONS:
        removed = tuple(model.REMOVED_GEMM_STAGES) + tuple(model.REMOVED_VECTOR_STAGES)
        fused_time, fused_traffic = common.fused_stage_time(
            model.build_frontier(dec), smem, tensor_roof, cuda_roof
        )
        removed_time = np.zeros_like(total_time)
        removed_traffic = np.zeros_like(total_time)
        for n in removed:
            removed_time = removed_time + stage_times[n]
            removed_traffic = removed_traffic + (
                gemm_min[n] if n in gemm_min else float(vec_traffic[n])
            )
        with np.errstate(invalid="ignore"):
            total_time_fused = total_time - removed_time + fused_time
        total_hbm_fused = total_hbm_baseline - removed_traffic + fused_traffic
        fi = int(np.nanargmin(total_time_fused))
        rows.append({
            "batch": batch,
            "tokens_per_expert": int(dec.TOKENS_PER_EXPERT),
            "attn_frac": attn_frac,
            "fusion": label,
            "unfused_time_ms": unfused_time * 1e3,
            "fused_time_ms": float(total_time_fused[fi]) * 1e3,
            "dtime_pct": (unfused_time - float(total_time_fused[fi])) / unfused_time * 100.0,
            "unfused_cuda": int(cuda[bi]),
            "fused_cuda": int(cuda[fi]),
            "dcuda": int(cuda[bi]) - int(cuda[fi]),
            "unfused_tensor": int(tensor[bi]),
            "fused_tensor": int(tensor[fi]),
            "dhbm_gib": float(total_hbm_baseline[bi] - total_hbm_fused[fi]) / 2**30,
        })
    return rows


def summarize(rows) -> None:
    from collections import defaultdict
    by_batch = defaultdict(list)
    for r in rows:
        by_batch[r["batch"]].append(r)
    print(f"\n{'batch':>7} {'tok/exp':>7} {'attn%':>6} | "
          f"{'best dtime% (fusion)':>24} | {'max |Δsplit| (fusion)':>24} | {'best-time fusion':>18}")
    for b in BATCHES:
        g = by_batch[b]
        best_dt = max(g, key=lambda r: r["dtime_pct"])
        best_dc = max(g, key=lambda r: abs(r["dcuda"]))
        star = "  *" if b == DEFAULT_BATCH else ""
        print(f"{b:>7}{star:>3} {g[0]['tokens_per_expert']:>5} {g[0]['attn_frac']*100:>5.1f}% | "
              f"{best_dt['dtime_pct']:+7.3f}% ({best_dt['fusion'].split()[0]:>3}) | "
              f"{best_dc['dcuda']:+5d} ({best_dc['fusion'].split()[0]:>3}) | "
              f"{best_dt['fusion']:>18}")

    # Per-fusion Δsplit and Δtime across batch.
    fusions = [f for f, _ in FUSIONS]
    for metric, fmt in [("dcuda", "{:+5d}"), ("dtime_pct", "{:+6.2f}")]:
        print(f"\n--- {metric} by batch x fusion ---")
        print("batch".rjust(7) + " | " + " ".join(f.split()[0].rjust(6) for f in fusions))
        for b in BATCHES:
            g = {r["fusion"]: r for r in by_batch[b]}
            cells = [fmt.format(g[f][metric]) for f in fusions]
            print(f"{b:>7} | " + " ".join(c.rjust(6) for c in cells))


def main() -> None:
    rows = []
    for b in BATCHES:
        rows += eval_batch(b)
    # Restore default config.
    dec.configure(DEFAULT_BATCH, SEQ_LEN)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "batch_sweep.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    summarize(rows)
    print(f"\nWrote {len(rows)} rows to {result_dir / 'batch_sweep.json'}")


if __name__ == "__main__":
    main()
