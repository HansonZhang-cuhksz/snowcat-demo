"""Hardware sensitivity-to-fusion sweep.

For decode and prefill, sweep four GPU performance knobs one at a time (HBM bandwidth, HBM
latency, tensor-core GFLOP/s, CUDA-core GFLOP/s) and, at each setting, measure how much
each of the six fusions changes the layer vs the unfused baseline:

  * dtime%   -- (unfused_time - fused_time) / unfused_time  (fusion's time effect)
  * dcuda    -- unfused_optimal_CUDA - fused_optimal_CUDA   (die-partition shift, cores)
  * dhbm     -- unfused_HBM - fused_HBM  (GiB; min-traffic convention)

"Sensitive to fusion" = a setup where the best fusion produces a large dtime% and/or moves
the optimal split. Everything is compared to the default (2.04 TB/s, 500 cyc, 512 / 5.64
GFLOP/s tensor/CUDA).

Efficiency: the Snowcat traffic frontiers are hardware-independent (they depend only on
GEMM dims + SMEM), so we build them once per stage and cache them; each hardware setting
then only re-runs the (fast, vectorised) time computation.

Run: conda run -n fusion python -m fusion.sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import decode_area_latency as dec
import prefill_area_latency as pre

# Coarser area grid than the 0.001 default: the sweep needs the optimal split + time, which
# is robust to grid resolution, and this is ~4x faster per hardware setup.
SWEEP_AREA_GRID_STEP = 0.002

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

# Default hardware (matches the reports).
DEF_BW = 2.04e12          # byte/s
DEF_LAT = 500             # HBM latency cycles
DEF_TF = 512e9            # tensor GFLOP/s/core
DEF_CF = 5.64e9           # CUDA GFLOP/s/core

# Single-parameter sweeps (values from the existing sensitivity tables).
SWEEPS = {
    "HBM bandwidth (TB/s)": ("bw", [1.02e12, 1.53e12, 2.04e12, 3.06e12, 4.08e12], 1e12),
    "HBM latency (cycles)": ("lat", [250, 500, 1000, 2000, 4000], 1),
    "Tensor GFLOP/s/core": ("tf", [256e9, 384e9, 512e9, 768e9, 1024e9], 1e9),
    "CUDA GFLOP/s/core": ("cf", [2.82e9, 4.23e9, 5.64e9, 8.46e9, 11.28e9], 1e9),
}


def set_hw(baseline, bw, lat, tf, cf) -> None:
    baseline.bw = bw
    baseline.HBM_LATENCY_CYCLES = lat
    baseline.TENSOR_FLOPS = tf
    baseline.ACTIVATION_FLOPS_PER_CUDA_CORE = cf
    # fusion.common carries its own copies (guarded); keep them in sync.
    common.bw = bw
    common.HBM_LATENCY_CYCLES = lat
    common.LATENCY_SECONDS = lat / baseline.CUDA_CLOCK_HZ


def install_frontier_cache(baseline) -> None:
    """Build the (hardware-independent) frontiers once, then reuse them every call."""
    cache: dict = {}
    orig = baseline.build_frontiers

    def cached(task_groups):
        if "frontiers" not in cache:
            cache["frontiers"] = orig(task_groups)[0]
        return cache["frontiers"], "cached"

    baseline.build_frontiers = cached


def eval_setup(baseline, bw, lat, tf, cf) -> list[dict]:
    set_hw(baseline, bw, lat, tf, cf)
    base = baseline.evaluate_layer()
    bi = base["best_index"]
    cuda = base["cuda_cores"]
    tensor = base["tensor_cores"]
    smem = base["smem_bytes"]
    tensor_roof = base["tensor_roof"]
    cuda_roof = base["cuda_roof"]
    total_time = base["total_time"]
    unfused_time = float(total_time[bi])

    # Baseline HBM (min-traffic convention), computed ONCE for this hardware setup: a per
    # aggregate-GEMM min-traffic map + the fixed vector/attn traffic. Reused for all fusions.
    stage_times = common.baseline_stage_times(base)
    vec_traffic = common._vector_stage_traffic(baseline)
    agg = base["aggregate_names"]
    gw = base["group_weights"]
    gemm_min: dict[str, np.ndarray] = {}
    for fr in base["frontiers"]:
        a = agg[fr.label]
        contrib = gw[fr.label] * common.baseline_gemm_min_traffic(fr, smem, tensor_roof)
        gemm_min[a] = gemm_min.get(a, 0.0) + contrib
    total_hbm_baseline = sum(gemm_min.values()) + sum(float(v) for v in vec_traffic.values())

    rows = []
    for label, model in FUSIONS:
        removed = tuple(model.REMOVED_GEMM_STAGES) + tuple(model.REMOVED_VECTOR_STAGES)
        fused_time, fused_traffic = common.fused_stage_time(
            model.build_frontier(baseline), smem, tensor_roof, cuda_roof
        )
        removed_time = np.zeros_like(total_time)
        removed_traffic = np.zeros_like(total_time)
        for n in removed:
            removed_time = removed_time + stage_times[n]
            if n in gemm_min:
                removed_traffic = removed_traffic + gemm_min[n]
            else:
                removed_traffic = removed_traffic + float(vec_traffic[n])
        with np.errstate(invalid="ignore"):
            total_time_fused = total_time - removed_time + fused_time
        total_hbm_fused = total_hbm_baseline - removed_traffic + fused_traffic
        fi = int(np.nanargmin(total_time_fused))
        rows.append({
            "fusion": label,
            "unfused_time_ms": unfused_time * 1e3,
            "fused_time_ms": float(total_time_fused[fi]) * 1e3,
            "dtime_pct": (unfused_time - float(total_time_fused[fi])) / unfused_time * 100.0,
            "unfused_cuda": int(cuda[bi]),
            "fused_cuda": int(cuda[fi]),
            "dcuda": int(cuda[bi]) - int(cuda[fi]),
            "unfused_tensor": int(tensor[bi]),
            "fused_tensor": int(tensor[fi]),
            "unfused_hbm_gib": float(total_hbm_baseline[bi]) / 2**30,
            "fused_hbm_gib": float(total_hbm_fused[fi]) / 2**30,
            "dhbm_gib": float(total_hbm_baseline[bi] - total_hbm_fused[fi]) / 2**30,
        })
    return rows


def run_stage(name, baseline) -> list[dict]:
    baseline.AREA_GRID_STEP = SWEEP_AREA_GRID_STEP
    install_frontier_cache(baseline)
    # Warm the cache at default hardware.
    set_hw(baseline, DEF_BW, DEF_LAT, DEF_TF, DEF_CF)
    baseline.evaluate_layer()

    out = []
    for sweep_name, (key, values, _unit) in SWEEPS.items():
        for v in values:
            bw, lat, tf, cf = DEF_BW, DEF_LAT, DEF_TF, DEF_CF
            if key == "bw":
                bw = v
            elif key == "lat":
                lat = int(v)
            elif key == "tf":
                tf = v
            elif key == "cf":
                cf = v
            is_default = (bw == DEF_BW and lat == DEF_LAT and tf == DEF_TF and cf == DEF_CF)
            for row in eval_setup(baseline, bw, lat, tf, cf):
                row.update({
                    "stage": name, "sweep": sweep_name, "param_value": v,
                    "is_default": is_default,
                })
                out.append(row)
    # Restore defaults.
    set_hw(baseline, DEF_BW, DEF_LAT, DEF_TF, DEF_CF)
    return out


def summarize(rows) -> None:
    """Print, per (stage, sweep, value), the fusion sensitivity: best |dtime%| and max |dcuda|."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["stage"], r["sweep"], r["param_value"])].append(r)

    for stage in ("decode", "prefill"):
        print(f"\n================  {stage.upper()}  — fusion sensitivity vs hardware  ================")
        for sweep_name, (key, values, unit) in SWEEPS.items():
            print(f"\n--- {sweep_name} ---")
            print(f"{'value':>10} {'def?':>5} | best dtime% (fusion) | max |Δsplit| cores (fusion) | dHBM range GiB")
            for v in values:
                g = groups.get((stage, sweep_name, v), [])
                if not g:
                    continue
                best_dt = max(g, key=lambda r: abs(r["dtime_pct"]))
                best_dc = max(g, key=lambda r: abs(r["dcuda"]))
                dhbm_lo = min(r["dhbm_gib"] for r in g)
                dhbm_hi = max(r["dhbm_gib"] for r in g)
                is_def = "  *" if g[0]["is_default"] else ""
                print(f"{v/unit:>10.4g} {is_def:>5} | "
                      f"{best_dt['dtime_pct']:+6.3f}% ({best_dt['fusion'].split()[0]}) | "
                      f"{best_dc['dcuda']:+4d} ({best_dc['fusion'].split()[0]}) | "
                      f"[{dhbm_lo:+.1f}, {dhbm_hi:+.1f}]")


def main() -> None:
    rows = []
    rows += run_stage("decode", dec)
    rows += run_stage("prefill", pre)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "sensitivity.json"
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=1)

    summarize(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
