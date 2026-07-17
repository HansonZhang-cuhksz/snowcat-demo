"""Fusion-agnostic latency-aware roofline + Snowcat engine.

Shared by every ``fusion/<name>`` analysis.  Nothing here is specific to a particular
fusion: it takes a set of CODA fused-traffic tile points (which encode *which*
intermediate is kept on chip and *which* aux buffers/HBM appear), collapses them to the
Snowcat Pareto frontier keyed on the one-stage working set ``W = buffer_bytes``, and
evaluates the latency-aware time with a per-kernel software-pipeline depth
``num_stages`` (``C``).

Chip constants, ``bw``, ``HBM_LATENCY_CYCLES`` and ``CUDA_CLOCK_HZ`` are imported from
``decode_area_latency`` so all fusions share one source of truth with the baseline.

Model (see ``notes/latency_pipeline_model.md``):

    N = W = buffer_bytes                                  (one-stage working set)
    latency = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    C_best  = min(floor(S_total/W), ceil(bw*latency/W))   (smallest optimal num_stages)
    BW_eff  = min(bw, C_best*W/latency)
    time    = count * max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T/BW_eff)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import decode_area_latency as dal
from coda_fused_traffic import (
    FusedTraffic,
    _partial_accumulator_traffic,
    _run_count,
    _traffic_for_pairwise_output_tile,
    _traffic_for_standard_output_tile,
    divisors,
)
from coda_fused_register_accumulator_traffic import REGISTER_ACCUMULATOR_LOOP_ORDERS


# One source of truth: reuse the baseline's chip constants.
bw = dal.bw
HBM_LATENCY_CYCLES = dal.HBM_LATENCY_CYCLES
CUDA_CLOCK_HZ = dal.CUDA_CLOCK_HZ
LATENCY_SECONDS = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
BYTE_PER_ELEMENT = dal.BYTE_PER_ELEMENT


@dataclass(frozen=True)
class FusedFrontier:
    """Pareto frontier (keyed on one-stage working set W) of a fused kernel.

    ``tensor_operations`` / ``cuda_operations`` are the fused kernel's compute totals
    (per invocation); ``count`` is the batched-GEMM multiplicity (e.g. active experts).
    """

    label: str
    count: int
    tensor_operations: float
    cuda_operations: float
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray
    bm: np.ndarray
    bn: np.ndarray
    bk: np.ndarray
    loop_orders: tuple[tuple[str, str, str], ...]


def standard_tile_points(
    m: int,
    n: int,
    k: int,
    *,
    final_tile_bytes,
    aux_hbm_bytes,
    aux_buffer_bytes,
    bytes_per_element: int = BYTE_PER_ELEMENT,
    loop_orders: tuple[tuple[str, str, str], ...] = REGISTER_ACCUMULATOR_LOOP_ORDERS,
) -> list[FusedTraffic]:
    """CODA fused-traffic points for a standard (single M x N output) fused GEMM.

    ``final_tile_bytes(m0, n0)``  -> bytes written per output tile (the fused result).
    ``aux_hbm_bytes(m0, n0, mt, nt)`` -> extra HBM bytes for the fused epilogue/prologue
        (residual reads, gamma reads, partial-stat spills, ...).
    ``aux_buffer_bytes(m0, n0)`` -> extra on-chip working set the fusion needs.
    Register-accumulator loop orders (K innermost) by default: no partial-accumulator
    HBM spill, matching the decode model's ``USE_REGISTER_ACCUMULATOR_MAPPINGS``.
    """
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for n0 in divisors(n):
                mt, nt = m // m0, n // n0
                for loop_order in loop_orders:
                    points.append(
                        _traffic_for_standard_output_tile(
                            m=m,
                            n=n,
                            k=k,
                            m0=m0,
                            n0=n0,
                            k0=k0,
                            loop_order=loop_order,
                            bytes_per_element=bytes_per_element,
                            final_tile_bytes=final_tile_bytes(m0, n0),
                            aux_hbm_bytes=aux_hbm_bytes(m0, n0, mt, nt),
                            aux_buffer_bytes=aux_buffer_bytes(m0, n0),
                        )
                    )
    return points


def pairwise_tile_points(
    m: int,
    p: int,
    k: int,
    *,
    final_tile_bytes,
    aux_hbm_bytes,
    aux_buffer_bytes,
    bytes_per_element: int = BYTE_PER_ELEMENT,
    loop_orders: tuple[tuple[str, str, str], ...] = REGISTER_ACCUMULATOR_LOOP_ORDERS,
) -> list[FusedTraffic]:
    """CODA fused-traffic points for a pairwise gate/up (SwiGLU) fused GEMM.

    The GEMM produces ``2*p`` interleaved gate/up columns; the epilogue stores ``p``
    activated columns.  ``p = n // 2``.
    """
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for p0 in divisors(p):
                mt, pt = m // m0, p // p0
                for loop_order in loop_orders:
                    points.append(
                        _traffic_for_pairwise_output_tile(
                            m=m,
                            p=p,
                            k=k,
                            m0=m0,
                            p0=p0,
                            k0=k0,
                            loop_order=loop_order,
                            bytes_per_element=bytes_per_element,
                            final_tile_bytes=final_tile_bytes(m0, p0),
                            aux_hbm_bytes=aux_hbm_bytes(m0, p0, mt, pt),
                            aux_buffer_bytes=aux_buffer_bytes(m0, p0),
                        )
                    )
    return points


def widened_a_points(
    m: int,
    n: int,
    k: int,
    *,
    a_width_mult: int,
    final_tile_bytes,
    aux_hbm_bytes,
    aux_buffer_bytes,
    bytes_per_element: int = BYTE_PER_ELEMENT,
    loop_orders: tuple[tuple[str, str, str], ...] = REGISTER_ACCUMULATOR_LOOP_ORDERS,
) -> list[FusedTraffic]:
    """Standard-tile GEMM points where the A input is read ``a_width_mult`` * wider than K.

    Models a *prologue* fusion that consumes a wider input than the GEMM's contraction
    dim and reduces it on chip (e.g. SwiGLU into down: the down GEMM's K = INTERMEDIATE,
    but it reads the ``2*INTERMEDIATE`` gate+up tensor and applies SwiGLU on chip to form
    the activated A tiles).  ``a_width_mult`` scales the A-tile read/buffer accordingly.
    W reads, output write, and partial-accumulator traffic are the standard GEMM's.
    """
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for n0 in divisors(n):
                mt, nt, kt = m // m0, n // n0, k // k0
                a_tile = m0 * k0 * a_width_mult * bytes_per_element
                w_tile = k0 * n0 * bytes_per_element
                raw_output_tile = m0 * n0 * bytes_per_element
                output_tiles = mt * nt
                for lo in loop_orders:
                    extents = {"M": mt, "K": kt, "N": nt}
                    a_reads = _run_count(lo, extents, ("M", "K")) * a_tile
                    w_reads = _run_count(lo, extents, ("K", "N")) * w_tile
                    pr, pw = _partial_accumulator_traffic(
                        lo, extents, output_tiles, raw_output_tile
                    )
                    hbm = (
                        a_reads
                        + w_reads
                        + pr
                        + pw
                        + output_tiles * final_tile_bytes(m0, n0)
                        + aux_hbm_bytes(m0, n0, mt, nt)
                    )
                    buffer = a_tile + w_tile + raw_output_tile + aux_buffer_bytes(m0, n0)
                    points.append(FusedTraffic(buffer, hbm, m0, n0, k0, lo))
    return points


def build_fused_frontier(
    label: str,
    count: int,
    tensor_operations: float,
    cuda_operations: float,
    points: list[FusedTraffic],
    tile_filter=None,
) -> FusedFrontier:
    """Collapse CODA fused-traffic ``points`` to the Pareto frontier keyed on W.

    Mirrors ``decode_area_latency.build_traffic_frontier``: sort by ``buffer_bytes``
    ascending, keep the running-min ``hbm_bytes`` (min-attainable traffic at each
    capacity), collapse duplicate capacities, then keep only the strictly-improving
    Pareto steps.  ``tile_filter(bm, bn, bk) -> bool`` optionally restricts to
    tensor-core-feasible tiles.
    """
    if tile_filter is not None:
        points = [p for p in points if tile_filter(p.bm, p.bn, p.bk)]
    if not points:
        raise ValueError(f"no fused tiling for {label} (after tile filter)")

    pairs = sorted(
        (p.buffer_bytes, p.hbm_bytes, p.bm, p.bn, p.bk, p.loop_order) for p in points
    )

    f_buffer: list[int] = []
    f_traffic: list[int] = []
    f_bm: list[int] = []
    f_bn: list[int] = []
    f_bk: list[int] = []
    f_lo: list[tuple[str, str, str]] = []
    best: tuple[int, int, int, int, tuple[str, str, str]] | None = None

    for buffer_bytes, traffic_bytes, bm, bn, bk, loop_order in pairs:
        if best is None or traffic_bytes < best[0]:
            best = (traffic_bytes, bm, bn, bk, loop_order)
        if f_buffer and buffer_bytes == f_buffer[-1]:
            f_traffic[-1] = best[0]
            f_bm[-1] = best[1]
            f_bn[-1] = best[2]
            f_bk[-1] = best[3]
            f_lo[-1] = best[4]
        else:
            f_buffer.append(buffer_bytes)
            f_traffic.append(best[0])
            f_bm.append(best[1])
            f_bn.append(best[2])
            f_bk.append(best[3])
            f_lo.append(best[4])

    buffers = np.array(f_buffer, dtype=np.int64)
    traffic = np.array(f_traffic, dtype=np.int64)
    bm_a = np.array(f_bm, dtype=np.int64)
    bn_a = np.array(f_bn, dtype=np.int64)
    bk_a = np.array(f_bk, dtype=np.int64)
    improved = np.r_[True, traffic[1:] < traffic[:-1]]

    return FusedFrontier(
        label=label,
        count=count,
        tensor_operations=tensor_operations,
        cuda_operations=cuda_operations,
        buffer_bytes=buffers[improved],
        traffic_bytes=traffic[improved],
        bm=bm_a[improved],
        bn=bn_a[improved],
        bk=bk_a[improved],
        loop_orders=tuple(lo for keep, lo in zip(improved, f_lo) if keep),
    )


def fused_stage_time(
    frontier: FusedFrontier,
    s_total: np.ndarray,
    tensor_roof: np.ndarray,
    cuda_roof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-area-node fused-kernel time and total HBM traffic (count*T).

    Per Pareto point, solve the smallest optimal num_stages and take the min-time
    point.  Returns ``(time, count*traffic)`` so the traffic is the whole stage total
    (consistent with ``decode_area_latency`` which multiplies per-invocation traffic by
    ``count``).
    """
    n = len(s_total)
    time_best = np.full(n, np.inf, dtype=float)
    traffic_best = np.full(n, np.nan, dtype=float)

    tensor_time = np.full(n, np.inf, dtype=float)
    np.divide(frontier.tensor_operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    cuda_time = np.full(n, np.inf, dtype=float)
    np.divide(frontier.cuda_operations, cuda_roof, out=cuda_time, where=cuda_roof > 0)
    compute_time = np.maximum(tensor_time, cuda_time)

    for i in range(len(frontier.buffer_bytes)):
        w_i = float(frontier.buffer_bytes[i])
        t_i = float(frontier.traffic_bytes[i])
        c_max = np.floor(s_total / w_i)
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * LATENCY_SECONDS / w_i))
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w_i / LATENCY_SECONDS)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem_time = t_i / bw_eff
        time_i = frontier.count * np.maximum(compute_time, mem_time)
        time_i = np.where(valid, time_i, np.inf)
        # Lexicographic (time, then traffic): among time-optimal tilings report the
        # MINIMUM traffic. This makes HBM well-defined for compute-bound kernels (where
        # many tilings tie on time); for memory-bound kernels the time-min tiling is
        # already unique so this changes nothing.
        equal = np.isclose(time_i, time_best, rtol=1e-9, atol=0.0)
        with np.errstate(invalid="ignore"):
            lower_traffic = t_i < traffic_best
        better = (time_i < time_best) | (equal & lower_traffic)
        time_best = np.where(better, time_i, time_best)
        traffic_best = np.where(better, t_i, traffic_best)

    return time_best, frontier.count * traffic_best


def select_mapping(
    frontier: FusedFrontier,
    s_total: float,
    tensor_roof: float,
    cuda_roof: float,
) -> dict[str, object] | None:
    """Winning tiling + num_stages of the fused kernel at one (fixed) SMEM budget."""
    best: dict[str, object] | None = None
    tensor_time = frontier.tensor_operations / tensor_roof if tensor_roof > 0 else np.inf
    cuda_time = frontier.cuda_operations / cuda_roof if cuda_roof > 0 else np.inf
    compute_time = max(tensor_time, cuda_time)
    for i in range(len(frontier.buffer_bytes)):
        w_i = int(frontier.buffer_bytes[i])
        t_i = int(frontier.traffic_bytes[i])
        c_max = int(s_total // w_i)
        if c_max < 1:
            continue
        c_sat = int(np.ceil(bw * LATENCY_SECONDS / w_i))
        c_best = min(c_max, c_sat)
        bw_eff = min(bw, c_best * w_i / LATENCY_SECONDS)
        mem_time = t_i / bw_eff
        time_i = frontier.count * max(compute_time, mem_time)
        # Lexicographic (time, then traffic): among time-optimal tilings pick min traffic.
        is_better = best is None or (
            time_i < best["time"] - 1e-9 * max(abs(best["time"]), 1.0)
            or (
                abs(time_i - best["time"]) <= 1e-9 * max(abs(best["time"]), 1.0)
                and t_i < best["traffic"]
            )
        )
        if is_better:
            if mem_time >= tensor_time and mem_time >= cuda_time:
                bottleneck = "memory"
            elif tensor_time >= cuda_time:
                bottleneck = "tensor"
            else:
                bottleneck = "cuda"
            best = {
                "bm": int(frontier.bm[i]),
                "bn": int(frontier.bn[i]),
                "bk": int(frontier.bk[i]),
                "loop_order": frontier.loop_orders[i],
                "num_stages": c_best,
                "max_feasible_stages": c_max,
                "one_stage_smem": w_i,
                "traffic": t_i,
                "oi": (frontier.tensor_operations + frontier.cuda_operations) / t_i,
                "bw_eff": bw_eff,
                "tensor_time": tensor_time,
                "cuda_time": cuda_time,
                "mem_time": mem_time,
                "time": time_i,
                "bottleneck": bottleneck,
            }
    return best


def format_mapping(
    frontier: FusedFrontier,
    s_total: float,
    tensor_roof: float,
    cuda_roof: float,
) -> str:
    m = select_mapping(frontier, s_total, tensor_roof, cuda_roof)
    if m is None:
        return "no fused mapping fits selected SMEM capacity"
    return (
        f"BM={m['bm']}, BN={m['bn']}, BK={m['bk']}, "
        f"loop_order={'-'.join(m['loop_order'])}, "
        f"num_stages={m['num_stages']} (max_feasible={m['max_feasible_stages']}), "
        f"one_stage_smem={m['one_stage_smem'] / 2**10:.3f} KiB, "
        f"traffic={m['traffic'] / 2**20:.3f} MiB, "
        f"OI={m['oi']:.6f} FLOP/byte, "
        f"BW_eff={m['bw_eff'] / 1e12:.6f} TB/s, "
        f"bottleneck={m['bottleneck']} "
        f"(tensor={m['tensor_time'] * 1e3:.4f} ms, "
        f"cuda={m['cuda_time'] * 1e3:.4f} ms, mem={m['mem_time'] * 1e3:.4f} ms)"
    )


def tensor_core_tile_allowed(bm: int, bn: int, bk: int) -> bool:
    return dal.tensor_core_tile_allowed(bm, bn, bk)


# --------------------------------------------------------------------------------------
# Orchestration: swap one fused kernel into the full decode-layer baseline and compare.
# --------------------------------------------------------------------------------------

def _vector_stage_traffic(baseline) -> dict[str, float]:
    """Scalar HBM traffic (constant across the area grid) of the non-GEMM stages."""
    return {
        "pre_attention_rmsnorm": baseline.PRE_ATTENTION_RMSNORM_TASK.traffic_bytes,
        "mla_attention": baseline.ATTENTION_CORE_TASK.traffic_bytes,
        "post_attention_residual_add": baseline.POST_ATTENTION_RESIDUAL_ADD_TASK.traffic_bytes,
        # Guard with INCLUDE_RMSNORM to match total_hbm_traffic_bytes() and the ops path.
        "rmsnorm_square_reduction": (
            baseline.RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes
            if baseline.INCLUDE_RMSNORM
            else 0.0
        ),
        "activation": baseline.ACTIVATION_TASK.traffic_bytes,
        "expert_weighted_sum": baseline.EXPERT_WEIGHTED_SUM_TASK.traffic_bytes,
        "residual_add": baseline.RESIDUAL_ADD_TASK.traffic_bytes,
    }


def _vector_stage_operations(baseline) -> dict[str, float]:
    return {
        "pre_attention_rmsnorm": baseline.PRE_ATTENTION_RMSNORM_TASK.operations,
        "mla_attention": baseline.ATTENTION_CORE_TASK.operations,
        "post_attention_residual_add": baseline.POST_ATTENTION_RESIDUAL_ADD_TASK.operations,
        "rmsnorm_square_reduction": (
            baseline.RMSNORM_SQUARE_REDUCTION_TASK.operations
            if baseline.INCLUDE_RMSNORM
            else 0.0
        ),
        "activation": baseline.ACTIVATION_TASK.operations,
        "expert_weighted_sum": baseline.EXPERT_WEIGHTED_SUM_TASK.operations,
        "residual_add": baseline.RESIDUAL_ADD_TASK.operations,
    }


def baseline_stage_times(base: dict) -> dict[str, np.ndarray]:
    """Map every removable baseline stage name -> its per-node time array."""
    times: dict[str, np.ndarray] = dict(base["task_times"])
    times["pre_attention_rmsnorm"] = base["pre_attention_rmsnorm_time"]
    times["mla_attention"] = base["attention_time"]
    times["post_attention_residual_add"] = base["post_attention_residual_add_time"]
    times["rmsnorm_square_reduction"] = base["rmsnorm_time"]
    times["activation"] = base["activation_time"]
    times["expert_weighted_sum"] = base["expert_weighted_sum_time"]
    times["residual_add"] = base["residual_add_time"]
    return times


def baseline_stage_traffic(baseline, base: dict) -> dict[str, np.ndarray]:
    """Map every removable baseline stage name -> its per-node HBM traffic array."""
    grid_len = len(base["rc"])
    traffic: dict[str, np.ndarray] = dict(base["task_traffic"])
    for name, value in _vector_stage_traffic(baseline).items():
        traffic[name] = np.full(grid_len, float(value))
    return traffic


def baseline_stage_operations(baseline, base: dict) -> dict[str, float]:
    ops: dict[str, float] = dict(baseline.task_operations_by_name)  # GEMM aggregates
    ops.update(_vector_stage_operations(baseline))
    return ops


def baseline_gemm_min_traffic(frontier, s_total: np.ndarray, tensor_roof: np.ndarray) -> np.ndarray:
    """Min traffic among time-optimal tilings for a baseline (tensor-only) GEMM stage.

    ``frontier`` is a decode/prefill ``TrafficFrontier`` (attrs ``operations``,
    ``count``, ``buffer_bytes``, ``traffic_bytes``).  The TIME matches the baseline
    module's ``gemm_time_from_frontier``; the traffic is the minimum among tilings that
    achieve it -- well-defined for compute-bound kernels (where the baseline module would
    otherwise report an arbitrary tied tiling).  Returns ``count * traffic`` per node.
    """
    n = len(s_total)
    time_best = np.full(n, np.inf, dtype=float)
    traffic_best = np.full(n, np.nan, dtype=float)
    tensor_time = np.full(n, np.inf, dtype=float)
    np.divide(frontier.operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    for i in range(len(frontier.buffer_bytes)):
        w = float(frontier.buffer_bytes[i])
        t = float(frontier.traffic_bytes[i])
        c_max = np.floor(s_total / w)
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * LATENCY_SECONDS / w))
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w / LATENCY_SECONDS)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem = t / bw_eff
        time_i = frontier.count * np.maximum(tensor_time, mem)
        time_i = np.where(valid, time_i, np.inf)
        equal = np.isclose(time_i, time_best, rtol=1e-9, atol=0.0)
        with np.errstate(invalid="ignore"):
            lower = t < traffic_best
        better = (time_i < time_best) | (equal & lower)
        time_best = np.where(better, time_i, time_best)
        traffic_best = np.where(better, t, traffic_best)
    return frontier.count * traffic_best


def layer_total_and_removed_traffic(baseline, base, removed_stages):
    """Recompute layer HBM under the min-traffic-among-time-optimal convention.

    Returns ``(total_hbm_baseline, removed_traffic)`` arrays.  All GEMM stages use
    ``baseline_gemm_min_traffic`` (consistent for compute-bound kernels); the non-GEMM
    vector/attention stages use their fixed traffic.  ``removed_traffic`` is the portion
    attributable to the fused kernel's removed stages.
    """
    smem_bytes = base["smem_bytes"]
    tensor_roof = base["tensor_roof"]
    n = len(smem_bytes)
    fixed = _vector_stage_traffic(baseline)              # name -> scalar
    aggregate_names = base["aggregate_names"]
    group_weights = base["group_weights"]

    total = np.zeros(n, dtype=float)
    removed_traffic = np.zeros(n, dtype=float)

    # GEMM stages (min-traffic convention).
    for frontier in base["frontiers"]:
        agg = aggregate_names[frontier.label]
        weight = group_weights[frontier.label]
        stage_traffic = weight * baseline_gemm_min_traffic(frontier, smem_bytes, tensor_roof)
        total = total + stage_traffic
        if agg in removed_stages:
            removed_traffic = removed_traffic + stage_traffic

    # Non-GEMM stages (fixed traffic).
    for name, value in fixed.items():
        total = total + float(value)
        if name in removed_stages:
            removed_traffic = removed_traffic + float(value)

    return total, removed_traffic


@dataclass
class FusionComparison:
    base: dict
    fused_frontier: FusedFrontier
    removed_stages: tuple[str, ...]
    fused_time: np.ndarray
    fused_traffic: np.ndarray
    total_time_fused: np.ndarray
    total_hbm_fused: np.ndarray
    total_hbm_baseline: np.ndarray
    removed_traffic: np.ndarray
    best_index_fused: int
    fused_operations: float
    removed_operations: float


def swap_fused_kernel(
    baseline,
    base: dict,
    fused_frontier: FusedFrontier,
    removed_stages: tuple[str, ...],
) -> FusionComparison:
    """Replace ``removed_stages`` in the baseline layer with the fused kernel.

    ``total_fused = total_baseline - sum(removed stage times) + fused_kernel_time``.
    Also recomputes the layer's total HBM traffic the same way, and reports FLOP
    accounting (fusion changes traffic; some fusions add redundant compute).
    ``baseline`` is the decode or prefill module (identical interface).
    """
    smem_bytes = base["smem_bytes"]
    tensor_roof = base["tensor_roof"]
    cuda_roof = base["cuda_roof"]

    fused_time, fused_traffic = fused_stage_time(
        fused_frontier, smem_bytes, tensor_roof, cuda_roof
    )

    stage_times = baseline_stage_times(base)
    stage_ops = baseline_stage_operations(baseline, base)

    removed_time = np.zeros(len(smem_bytes), dtype=float)
    removed_ops = 0.0
    for name in removed_stages:
        if name not in stage_times:
            raise KeyError(f"removed stage '{name}' not found in baseline stages")
        removed_time = removed_time + stage_times[name]
        removed_ops += stage_ops.get(name, 0.0)

    # HBM under the min-traffic-among-time-optimal convention (well-defined even when the
    # kernels are compute-bound, i.e. prefill). removed_traffic is the removed stages'
    # share; fused_traffic already uses the same convention (fused_stage_time lexicographic).
    total_hbm_baseline, removed_traffic = layer_total_and_removed_traffic(
        baseline, base, removed_stages
    )
    total_hbm_fused = total_hbm_baseline - removed_traffic + fused_traffic

    # Invalid (too-small-SMEM) nodes carry inf in both base and fused; inf-inf -> nan is
    # expected there and those nodes are dropped by nanargmin. Silence the noise.
    with np.errstate(invalid="ignore"):
        total_time_fused = base["total_time"] - removed_time + fused_time
    best_index_fused = int(np.nanargmin(total_time_fused))

    # count-scaled to compare against removed_operations (which are stage totals).
    fused_ops = fused_frontier.count * (
        fused_frontier.tensor_operations + fused_frontier.cuda_operations
    )

    return FusionComparison(
        base=base,
        fused_frontier=fused_frontier,
        removed_stages=removed_stages,
        fused_time=fused_time,
        fused_traffic=fused_traffic,
        total_time_fused=total_time_fused,
        total_hbm_fused=total_hbm_fused,
        total_hbm_baseline=total_hbm_baseline,
        removed_traffic=removed_traffic,
        best_index_fused=best_index_fused,
        fused_operations=fused_ops,
        removed_operations=removed_ops,
    )


def print_comparison(baseline, cmp: FusionComparison, title: str) -> None:
    base = cmp.base
    rc, rt, r_smem = base["rc"], base["rt"], base["r_smem"]
    smem_bytes = base["smem_bytes"]
    cuda_cores, tensor_cores = base["cuda_cores"], base["tensor_cores"]
    tensor_roof, cuda_roof = base["tensor_roof"], base["cuda_roof"]
    modeled_ops = base["modeled_operations"]

    bi = base["best_index"]
    fi = cmp.best_index_fused

    print(f"\n================  {title}  ================")
    print(f"Fused kernel: {cmp.fused_frontier.label} "
          f"(count={cmp.fused_frontier.count})")
    print(f"Replaces baseline stages: {', '.join(cmp.removed_stages)}")

    print("\n--- FLOP accounting (fusion changes traffic; compute may add redundancy) ---")
    print(f"fused kernel FLOPs:   {cmp.fused_operations / 1e9:.6f} GFLOP")
    print(f"removed stage FLOPs:  {cmp.removed_operations / 1e9:.6f} GFLOP")
    delta = cmp.fused_operations - cmp.removed_operations
    rel = abs(delta) / max(cmp.removed_operations, 1.0)
    if rel < 1e-9:
        print("compute: conserved (pure traffic fusion)")
    else:
        print(f"compute: {delta / 1e9:+.6f} GFLOP "
              f"({rel * 100:+.4f}%) redundant/extra compute introduced by the fusion")

    def _fmt_opt(idx, total_time, total_hbm):
        return (
            f"  rc={rc[idx]:.6g}  rt={rt[idx]:.6g}  r_smem={r_smem[idx]:.6g}\n"
            f"  SMEM={smem_bytes[idx] / 2**20:.3f} MiB  "
            f"CUDA={int(cuda_cores[idx])}  Tensor={int(tensor_cores[idx])}\n"
            f"  total time={total_time[idx] * 1e3:.6f} ms  "
            f"total HBM={total_hbm[idx] / 2**30:.3f} GiB  "
            f"throughput={modeled_ops / total_time[idx] / 1e12:.3f} TFLOP/s"
        )

    print("\n--- Unfused baseline optimum ---")
    print(_fmt_opt(bi, base["total_time"], cmp.total_hbm_baseline))
    print("\n--- Fused optimum ---")
    print(_fmt_opt(fi, cmp.total_time_fused, cmp.total_hbm_fused))

    dt = (base["total_time"][bi] - cmp.total_time_fused[fi]) * 1e3
    dpct = dt / (base["total_time"][bi] * 1e3) * 100.0
    dh = (cmp.total_hbm_baseline[bi] - cmp.total_hbm_fused[fi]) / 2**20
    print("\n--- Layer-level effect of the fusion (fused optimum vs baseline optimum) ---")
    print(f"  total time:  {dt:+.6f} ms  ({dpct:+.4f}%)")
    print(f"  total HBM:   {dh:+.3f} MiB")

    # Kernel-level effect at the fused optimum: fused kernel vs the removed stages.
    stage_times = baseline_stage_times(base)
    removed_time_fi = sum(stage_times[n][fi] for n in cmp.removed_stages)
    removed_traffic_fi = cmp.removed_traffic[fi]  # min-traffic convention (see swap_fused_kernel)
    print("\n--- Kernel-level effect at the fused optimum ---")
    print(f"  removed stages time:  {removed_time_fi * 1e3:.6f} ms  "
          f"HBM: {removed_traffic_fi / 2**20:.3f} MiB")
    print(f"  fused kernel time:    {cmp.fused_time[fi] * 1e3:.6f} ms  "
          f"HBM: {cmp.fused_traffic[fi] / 2**20:.3f} MiB")
    print(f"  kernel time saved:    {(removed_time_fi - cmp.fused_time[fi]) * 1e3:+.6f} ms")
    print(f"  kernel HBM saved:     "
          f"{(removed_traffic_fi - cmp.fused_traffic[fi]) / 2**20:+.3f} MiB")
    print("  fused mapping: "
          + format_mapping(cmp.fused_frontier, smem_bytes[fi], tensor_roof[fi], cuda_roof[fi]))


def save_plots(baseline, cmp: FusionComparison, result_dir, title: str, prefix: str = "") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from pathlib import Path

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    base = cmp.base
    rc, rt = base["rc"], base["rt"]
    paths: list[str] = []

    # (1) fused total-time area map
    valid = np.isfinite(cmp.total_time_fused) & (cmp.total_time_fused > 0)
    fig = plt.figure(figsize=(9, 6.5))
    tms = cmp.total_time_fused[valid] * 1e3
    sc = plt.scatter(rt[valid], rc[valid], c=tms, s=7, cmap="viridis_r",
                     norm=LogNorm(vmin=tms.min(), vmax=tms.max()))
    plt.colorbar(sc, label="Fused total layer time (ms)")
    fi = cmp.best_index_fused
    plt.scatter([rt[fi]], [rc[fi]], marker="*", s=220, edgecolor="k",
                facecolor="white", label="fused optimum", zorder=5)
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title(f"{title}\nLayer time across area split (only this kernel fused)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    p = str(result_dir / f"{prefix}total_time_area.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # (2) comparison bars: layer-level (near-null) and kernel-level (the real saving)
    bi = base["best_index"]
    stage_times = baseline_stage_times(base)
    rm_time = sum(stage_times[n][fi] for n in cmp.removed_stages) * 1e3
    rm_hbm = cmp.removed_traffic[fi] / 2**20
    fu_time = cmp.fused_time[fi] * 1e3
    fu_hbm = cmp.fused_traffic[fi] / 2**20

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    labels = ["total time (ms)", "total HBM (GiB)"]
    base_vals = [base["total_time"][bi] * 1e3, cmp.total_hbm_baseline[bi] / 2**30]
    fused_vals = [cmp.total_time_fused[fi] * 1e3, cmp.total_hbm_fused[fi] / 2**30]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, base_vals, 0.4, label="unfused baseline")
    ax.bar(x + 0.2, fused_vals, 0.4, label="fused")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Layer level (at each design's optimum)")
    ax.legend()
    for i, (b, f) in enumerate(zip(base_vals, fused_vals)):
        ax.text(i - 0.2, b, f"{b:.4g}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, f, f"{f:.4g}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    labels2 = ["kernel time (ms)", "kernel HBM (MiB)"]
    rm_vals = [rm_time, rm_hbm]
    fu_vals = [fu_time, fu_hbm]
    x = np.arange(len(labels2))
    ax.bar(x - 0.2, rm_vals, 0.4, label="unfused (removed stages)")
    ax.bar(x + 0.2, fu_vals, 0.4, label="fused kernel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels2)
    ax.set_title("Fused-kernel level (at fused optimum)")
    ax.legend()
    for i, (b, f) in enumerate(zip(rm_vals, fu_vals)):
        ax.text(i - 0.2, b, f"{b:.4g}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, f, f"{f:.4g}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    p = str(result_dir / f"{prefix}fusion_comparison.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    return paths


def run_fusion(
    model, baseline, title: str, result_dir, csv_name: str, prefix: str = ""
) -> "FusionComparison":
    """End-to-end driver used by every ``fusion/<name>/analysis*.py``.

    ``model`` must expose ``build_frontier(baseline)`` -> FusedFrontier, and the tuples
    ``REMOVED_GEMM_STAGES`` / ``REMOVED_VECTOR_STAGES`` naming the baseline stages the
    fused kernel replaces.  ``baseline`` is the decode or prefill module (same interface);
    ``prefix`` disambiguates the plot filenames (e.g. ``"prefill_"``).
    """
    # The roofline chip constants (bw, latency, bytes/elem) are module-level, sourced from
    # decode_area_latency. They are identical in prefill_area_latency; guard loudly so a
    # future divergence fails fast instead of silently mis-modelling the injected baseline.
    for name, ours, theirs in (
        ("bw", bw, baseline.bw),
        ("HBM_LATENCY_CYCLES", HBM_LATENCY_CYCLES, baseline.HBM_LATENCY_CYCLES),
        ("CUDA_CLOCK_HZ", CUDA_CLOCK_HZ, baseline.CUDA_CLOCK_HZ),
        ("BYTE_PER_ELEMENT", BYTE_PER_ELEMENT, baseline.BYTE_PER_ELEMENT),
    ):
        if ours != theirs:
            raise ValueError(
                f"baseline {baseline.__name__} {name}={theirs} != fusion.common {name}={ours}; "
                f"thread the constant through instead of using the decode-sourced module global"
            )

    base = baseline.evaluate_layer()
    fused_frontier = model.build_frontier(baseline)
    removed = tuple(model.REMOVED_GEMM_STAGES) + tuple(model.REMOVED_VECTOR_STAGES)
    cmp = swap_fused_kernel(baseline, base, fused_frontier, removed)
    print_comparison(baseline, cmp, title)
    plots = save_plots(baseline, cmp, result_dir, title, prefix)
    csv_path = write_sweep_csv(cmp, result_dir, csv_name)
    print("\n--- Outputs ---")
    for p in plots:
        print(f"  plot: {p}")
    print(f"  csv:  {csv_path}")
    return cmp


def write_sweep_csv(cmp: FusionComparison, result_dir, filename: str) -> str:
    import csv
    from pathlib import Path

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    base = cmp.base
    rc, rt, r_smem = base["rc"], base["rt"], base["r_smem"]
    smem_bytes = base["smem_bytes"]
    cuda_cores, tensor_cores = base["cuda_cores"], base["tensor_cores"]
    modeled_ops = base["modeled_operations"]
    path = str(result_dir / filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rc", "rt", "r_smem", "smem_mib", "cuda_cores", "tensor_cores",
            "total_time_fused_ms", "total_hbm_fused_mib", "effective_tflops",
            "fused_kernel_time_ms", "fused_kernel_hbm_mib",
        ])
        for i in range(len(rc)):
            w.writerow([
                f"{rc[i]:.6g}", f"{rt[i]:.6g}", f"{r_smem[i]:.6g}",
                f"{smem_bytes[i] / 2**20:.6f}", int(cuda_cores[i]), int(tensor_cores[i]),
                f"{cmp.total_time_fused[i] * 1e3:.6f}",
                f"{cmp.total_hbm_fused[i] / 2**20:.6f}",
                f"{modeled_ops / cmp.total_time_fused[i] / 1e12:.6f}",
                f"{cmp.fused_time[i] * 1e3:.6f}",
                f"{cmp.fused_traffic[i] / 2**20:.6f}",
            ])
    return path
