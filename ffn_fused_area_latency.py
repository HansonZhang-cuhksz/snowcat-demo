from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from coda_fused_traffic import (
    FusedTraffic,
    traffic_points_gemm_rms_scale,
    traffic_points_gemm_rms_swiglu,
)
from coda_fused_register_accumulator_traffic import (
    REGISTER_ACCUMULATOR_LOOP_ORDERS,
    traffic_points_gemm_rms_scale as traffic_points_gemm_rms_scale_register_accum,
    traffic_points_gemm_rms_swiglu as traffic_points_gemm_rms_swiglu_register_accum,
)
from expert_distribution import ExpertTokenDistribution, binomial_expert_token_distribution
from expert_workload import even_expert_token_split, padded_gemm_groups, padded_m
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


# Chip constants. Keep these aligned with ffn_area.py for apples-to-apples runs.
# A_total = 694.116 * 10**6             # um^2
A_total = 136.29 * 10**6               # um^2
A_bit = 0.0864                         # um^2/bit
logic_density = 39.98                  # MTr/mm^2 == transistors/um^2
bw = 2.04 * 10**12                     # byte/s

num_sm = 1
A_total = A_total / num_sm
bw = bw / num_sm

CUDA_CORE_TRANSISTORS = 0.2 * 10**6     # transistors/CUDA core
TENSOR_CORE_TRANSISTORS = 6.0 * 10**6   # transistors/tensor core
TENSOR_FLOPS = 512 * 1.00 * 10**9       # flops/s/tensor core
CUDA_CLOCK_HZ = 1410 * 10**6            # Hz
ACTIVATION_FLOPS_PER_CUDA_CORE = 5.64 * 10**9  # flops/s/CUDA core

A_cuda_core = CUDA_CORE_TRANSISTORS / logic_density
A_tensor_core = TENSOR_CORE_TRANSISTORS / logic_density

# Decode FFN workload.
BYTE_PER_ELEMENT = 2
BATCH_TOKENS = 4096
EXPERTS = 256
ROUTER_TOP_K = 8
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
TENSOR_CORE_MIN_BM = 16
TENSOR_CORE_MIN_BN = 8
TENSOR_CORE_MIN_BK = 16

# Even token->expert distribution.  Handles small batches: ceil/floor token split when
# batch*top_k isn't a multiple of EXPERTS, reduced active-expert count when the batch is
# too small for every expert to get a token, and M padded to 16 for the per-expert GEMMs
# when tokens/expert < 16 (tensor core underutilized; the SwiGLU epilogue still runs on
# the real token count).
EVEN_SPLIT = even_expert_token_split(BATCH_TOKENS, EXPERTS, ROUTER_TOP_K)
TOKENS_PER_EXPERT = EVEN_SPLIT.floor_tokens  # nominal floor tokens/expert (reference)

SWIGLU_FLOPS_PER_ELEMENT = 8.0
INCLUDE_RMSNORM = True
AREA_GRID_STEP = 0.001

# False preserves the original fused Snowcat-style mapspace and output
# filenames.  True restricts fused GEMM mappings to M-N-K and N-M-K, where the
# output accumulator tile stays live through the K reduction.
USE_REGISTER_ACCUMULATOR_MAPPINGS = True

# False preserves the original even-routing estimate: every expert receives
# TOKENS_PER_EXPERT tokens.  True uses an expected-value random-routing model
# where each expert's token count follows Binomial(BATCH_TOKENS, top_k/EXPERTS).
USE_RANDOM_EXPERT_DISTRIBUTION = False
EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF = 1e-12

# Pipeline-depth / latency-hiding model.  num_stages (C) is solved per kernel:
# each concurrent task occupies one fused tile working set (W = buffer_bytes,
# GEMM tiles + epilogue aux) in SMEM, and C tasks stay in flight to hide HBM
# latency (BW_eff = min(bw, C*W/latency)).
HBM_LATENCY_CYCLES = 500
HBM_CLOCK_HZ = 1215 * 10**6

@dataclass(frozen=True)
class ReductionTask:
    name: str
    rows: int
    columns: int
    bytes_per_input: int
    bytes_per_output: int

    @property
    def operations(self) -> int:
        return self.rows * self.columns + self.rows * (self.columns - 1)

    @property
    def traffic_bytes(self) -> int:
        input_reads = self.rows * self.columns * self.bytes_per_input
        output_writes = self.rows * self.bytes_per_output
        return input_reads + output_writes

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class VectorTask:
    name: str
    elements: int
    count: int
    flops_per_element: float
    bytes_per_element_traffic: int

    @property
    def operations(self) -> float:
        return self.count * self.elements * self.flops_per_element

    @property
    def traffic_bytes(self) -> int:
        return self.count * self.elements * self.bytes_per_element_traffic

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class ExpertWeightedSumTask:
    name: str
    tokens: int
    top_k: int
    hidden_size: int
    bytes_per_activation: int
    bytes_per_weight: int
    bytes_per_output: int

    @property
    def operations(self) -> int:
        multiply_ops = self.tokens * self.top_k * self.hidden_size
        add_ops = self.tokens * (self.top_k - 1) * self.hidden_size
        return multiply_ops + add_ops

    @property
    def traffic_bytes(self) -> int:
        activation_reads = (
            self.tokens * self.top_k * self.hidden_size * self.bytes_per_activation
        )
        weight_reads = self.tokens * self.top_k * self.bytes_per_weight
        output_writes = self.tokens * self.hidden_size * self.bytes_per_output
        return activation_reads + weight_reads + output_writes

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class FusedGemmStage:
    name: str
    m: int
    n: int
    k: int
    count: int
    tensor_operations: int
    cuda_operations: float
    traffic_points_fn: Callable[[], list[FusedTraffic]]

    @property
    def operations(self) -> float:
        return self.tensor_operations + self.cuda_operations

    @property
    def total_operations(self) -> float:
        return self.count * self.operations


@dataclass(frozen=True)
class FusedTrafficFrontier:
    stage: FusedGemmStage
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray
    bm: np.ndarray
    bn: np.ndarray
    bk: np.ndarray
    loop_orders: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class StandardGemmStage:
    name: str
    m: int
    n: int
    k: int
    count: int

    @property
    def operations(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def total_operations(self) -> int:
        return self.count * self.operations


@dataclass(frozen=True)
class StandardTrafficFrontier:
    stage: StandardGemmStage
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray
    bm: np.ndarray
    bn: np.ndarray
    bk: np.ndarray
    loop_orders: tuple[tuple[str, str, str], ...]


RMSNORM_SQUARE_REDUCTION_TASK = ReductionTask(
    name="rmsnorm_square_reduction",
    rows=BATCH_TOKENS,
    columns=HIDDEN_SIZE,
    bytes_per_input=BYTE_PER_ELEMENT,
    bytes_per_output=4,
)

EXPERT_WEIGHTED_SUM_TASK = ExpertWeightedSumTask(
    name="expert_weighted_sum",
    tokens=BATCH_TOKENS,
    top_k=ROUTER_TOP_K,
    hidden_size=HIDDEN_SIZE,
    bytes_per_activation=BYTE_PER_ELEMENT,
    bytes_per_weight=BYTE_PER_ELEMENT,
    bytes_per_output=BYTE_PER_ELEMENT,
)

RESIDUAL_ADD_TASK = VectorTask(
    name="residual_add",
    elements=BATCH_TOKENS * HIDDEN_SIZE,
    count=1,
    flops_per_element=1.0,
    bytes_per_element_traffic=3 * BYTE_PER_ELEMENT,
)


def make_fused_stages() -> list[FusedGemmStage]:
    router_traffic_points_fn = (
        traffic_points_gemm_rms_scale_register_accum
        if USE_REGISTER_ACCUMULATOR_MAPPINGS
        else traffic_points_gemm_rms_scale
    )
    up_gate_traffic_points_fn = (
        traffic_points_gemm_rms_swiglu_register_accum
        if USE_REGISTER_ACCUMULATOR_MAPPINGS
        else traffic_points_gemm_rms_swiglu
    )

    # Router runs over all batch tokens (M = batch, padded to 16 for feasibility when
    # batch < 16); the RMS-scale epilogue uses the real batch row count.
    router_m = padded_m(BATCH_TOKENS, TENSOR_CORE_MIN_BM)
    router = FusedGemmStage(
        name="router_rms_scale",
        m=router_m,
        n=EXPERTS,
        k=HIDDEN_SIZE,
        count=1,
        tensor_operations=2 * router_m * EXPERTS * HIDDEN_SIZE,
        cuda_operations=BATCH_TOKENS * EXPERTS,
        traffic_points_fn=lambda: router_traffic_points_fn(
            router_m,
            EXPERTS,
            HIDDEN_SIZE,
            BYTE_PER_ELEMENT,
        ),
    )

    # One up_gate stage per distinct per-expert token count (ceil/floor split).  The
    # GEMM (tensor) and traffic use the padded M; the SwiGLU epilogue (cuda_operations)
    # uses the real token count.
    stages = [router]
    multi = len(EVEN_SPLIT.token_groups) > 1
    for tokens, count in EVEN_SPLIT.token_groups:
        m_pad = padded_m(tokens, TENSOR_CORE_MIN_BM)
        name = (
            f"up_gate_rms_swiglu_m{tokens}_x{count}"
            if multi
            else f"up_gate_rms_swiglu_x{count}"
        )
        stages.append(
            FusedGemmStage(
                name=name,
                m=m_pad,
                n=2 * INTERMEDIATE_SIZE,
                k=HIDDEN_SIZE,
                count=count,
                tensor_operations=2 * m_pad * (2 * INTERMEDIATE_SIZE) * HIDDEN_SIZE,
                cuda_operations=(
                    tokens * (2 * INTERMEDIATE_SIZE)
                    + tokens * INTERMEDIATE_SIZE * SWIGLU_FLOPS_PER_ELEMENT
                ),
                traffic_points_fn=lambda m_pad=m_pad: up_gate_traffic_points_fn(
                    m_pad,
                    2 * INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                    BYTE_PER_ELEMENT,
                ),
            )
        )

    return stages


def make_standard_stages() -> list[StandardGemmStage]:
    # down GEMM has no epilogue, so its cost depends only on the padded M; merge experts
    # by padded M.
    m_groups = padded_gemm_groups(EVEN_SPLIT, TENSOR_CORE_MIN_BM)
    multi = len(m_groups) > 1
    stages = []
    for m, count in m_groups:
        name = f"down_m{m}_x{count}" if multi else f"down_x{count}"
        stages.append(
            StandardGemmStage(
                name=name,
                m=m,
                n=HIDDEN_SIZE,
                k=INTERMEDIATE_SIZE,
                count=count,
            )
        )
    return stages


def expert_token_distribution() -> ExpertTokenDistribution:
    return binomial_expert_token_distribution(
        batch_tokens=BATCH_TOKENS,
        experts=EXPERTS,
        top_k=ROUTER_TOP_K,
        probability_cutoff=EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF,
    )


def make_random_expert_stages(
    distribution: ExpertTokenDistribution,
) -> tuple[
    list[FusedGemmStage],
    list[StandardGemmStage],
    dict[str, float],
    dict[str, str],
]:
    fused_stages = [make_fused_stages()[0]]
    standard_stages: list[StandardGemmStage] = []
    stage_weights = {fused_stages[0].name: 1.0}
    aggregate_names = {fused_stages[0].name: fused_stages[0].name}
    up_gate_traffic_points_fn = (
        traffic_points_gemm_rms_swiglu_register_accum
        if USE_REGISTER_ACCUMULATOR_MAPPINGS
        else traffic_points_gemm_rms_swiglu
    )

    for tokens, probability in distribution.support:
        if tokens == 0:
            continue

        # Pad M to the tensor-core minimum tile for the GEMM/traffic; the SwiGLU
        # epilogue (cuda_operations) uses the real token count.
        m_pad = padded_m(tokens, TENSOR_CORE_MIN_BM)
        up_gate_name = f"up_gate_rms_swiglu_m{tokens}"
        fused_stages.append(
            FusedGemmStage(
                name=up_gate_name,
                m=m_pad,
                n=2 * INTERMEDIATE_SIZE,
                k=HIDDEN_SIZE,
                count=1,
                tensor_operations=2 * m_pad * (2 * INTERMEDIATE_SIZE) * HIDDEN_SIZE,
                cuda_operations=(
                    tokens * (2 * INTERMEDIATE_SIZE)
                    + tokens * INTERMEDIATE_SIZE * SWIGLU_FLOPS_PER_ELEMENT
                ),
                traffic_points_fn=lambda m_pad=m_pad: up_gate_traffic_points_fn(
                    m_pad,
                    2 * INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                    BYTE_PER_ELEMENT,
                ),
            )
        )
        stage_weights[up_gate_name] = EXPERTS * probability
        aggregate_names[up_gate_name] = "up_gate_rms_swiglu_random_expected"

        down_name = f"down_m{tokens}"
        standard_stages.append(
            StandardGemmStage(
                name=down_name,
                m=m_pad,
                n=HIDDEN_SIZE,
                k=INTERMEDIATE_SIZE,
                count=1,
            )
        )
        stage_weights[down_name] = EXPERTS * probability
        aggregate_names[down_name] = "down_random_expected"

    return fused_stages, standard_stages, stage_weights, aggregate_names


def output_paths() -> tuple[str, str]:
    suffix_parts = []
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        suffix_parts.append("register_accumulator")
    if USE_RANDOM_EXPERT_DISTRIBUTION:
        suffix_parts.append("random_experts")
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    return (
        f"./result/ffn_fused_area_latency{suffix}_times.csv",
        f"./result/ffn_fused_area_latency{suffix}_total_time.png",
    )


def tensor_core_tile_allowed(bm: int, bn: int, bk: int) -> bool:
    return (
        bm >= TENSOR_CORE_MIN_BM
        and bn >= TENSOR_CORE_MIN_BN
        and bk >= TENSOR_CORE_MIN_BK
    )


def build_fused_frontier(stage: FusedGemmStage) -> FusedTrafficFrontier:
    points = [
        point
        for point in stage.traffic_points_fn()
        if tensor_core_tile_allowed(point.bm, point.bn, point.bk)
    ]
    if not points:
        raise ValueError(
            f"no tensor-core-compatible fused mapping for {stage.name}; "
            f"requires BM>={TENSOR_CORE_MIN_BM}, "
            f"BN>={TENSOR_CORE_MIN_BN}, BK>={TENSOR_CORE_MIN_BK}"
        )
    pairs = sorted(
        (
            point.buffer_bytes,
            point.hbm_bytes,
            point.bm,
            point.bn,
            point.bk,
            point.loop_order,
        )
        for point in points
    )

    frontier_buffer_list: list[int] = []
    frontier_traffic_list: list[int] = []
    frontier_bm_list: list[int] = []
    frontier_bn_list: list[int] = []
    frontier_bk_list: list[int] = []
    frontier_loop_order_list: list[tuple[str, str, str]] = []
    best: tuple[int, int, int, int, tuple[str, str, str]] | None = None

    for buffer_bytes, traffic_bytes, bm, bn, bk, loop_order in pairs:
        if best is None or traffic_bytes < best[0]:
            best = (traffic_bytes, bm, bn, bk, loop_order)
        if frontier_buffer_list and buffer_bytes == frontier_buffer_list[-1]:
            frontier_traffic_list[-1] = best[0]
            frontier_bm_list[-1] = best[1]
            frontier_bn_list[-1] = best[2]
            frontier_bk_list[-1] = best[3]
            frontier_loop_order_list[-1] = best[4]
        else:
            frontier_buffer_list.append(buffer_bytes)
            frontier_traffic_list.append(best[0])
            frontier_bm_list.append(best[1])
            frontier_bn_list.append(best[2])
            frontier_bk_list.append(best[3])
            frontier_loop_order_list.append(best[4])

    frontier_buffers = np.array(frontier_buffer_list, dtype=np.int64)
    frontier_traffic = np.array(frontier_traffic_list, dtype=np.int64)
    frontier_bm = np.array(frontier_bm_list, dtype=np.int64)
    frontier_bn = np.array(frontier_bn_list, dtype=np.int64)
    frontier_bk = np.array(frontier_bk_list, dtype=np.int64)
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return FusedTrafficFrontier(
        stage=stage,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
        bm=frontier_bm[improved],
        bn=frontier_bn[improved],
        bk=frontier_bk[improved],
        loop_orders=tuple(
            loop_order
            for keep, loop_order in zip(improved, frontier_loop_order_list)
            if keep
        ),
    )


def build_standard_frontier(stage: StandardGemmStage) -> StandardTrafficFrontier:
    workload = GemmWorkload(
        m=stage.m,
        k=stage.k,
        n=stage.n,
        bytes_per_element=BYTE_PER_ELEMENT,
    )
    pairs = sorted(
        (
            point.buffer_bytes,
            point.backing_store_bytes,
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
            point.mapping.loop_order,
        )
        for point in enumerate_mappings(workload)
        if tensor_core_tile_allowed(
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
        )
    )
    if not pairs:
        raise ValueError(
            f"no tensor-core-compatible standard mapping for {stage.name}; "
            f"requires BM>={TENSOR_CORE_MIN_BM}, "
            f"BN>={TENSOR_CORE_MIN_BN}, BK>={TENSOR_CORE_MIN_BK}"
        )

    frontier_buffer_list: list[int] = []
    frontier_traffic_list: list[int] = []
    frontier_bm_list: list[int] = []
    frontier_bn_list: list[int] = []
    frontier_bk_list: list[int] = []
    frontier_loop_order_list: list[tuple[str, str, str]] = []
    best: tuple[int, int, int, int, tuple[str, str, str]] | None = None

    for buffer_bytes, traffic_bytes, bm, bn, bk, loop_order in pairs:
        if best is None or traffic_bytes < best[0]:
            best = (traffic_bytes, bm, bn, bk, loop_order)
        if frontier_buffer_list and buffer_bytes == frontier_buffer_list[-1]:
            frontier_traffic_list[-1] = best[0]
            frontier_bm_list[-1] = best[1]
            frontier_bn_list[-1] = best[2]
            frontier_bk_list[-1] = best[3]
            frontier_loop_order_list[-1] = best[4]
        else:
            frontier_buffer_list.append(buffer_bytes)
            frontier_traffic_list.append(best[0])
            frontier_bm_list.append(best[1])
            frontier_bn_list.append(best[2])
            frontier_bk_list.append(best[3])
            frontier_loop_order_list.append(best[4])

    frontier_buffers = np.array(frontier_buffer_list, dtype=np.int64)
    frontier_traffic = np.array(frontier_traffic_list, dtype=np.int64)
    frontier_bm = np.array(frontier_bm_list, dtype=np.int64)
    frontier_bn = np.array(frontier_bn_list, dtype=np.int64)
    frontier_bk = np.array(frontier_bk_list, dtype=np.int64)
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return StandardTrafficFrontier(
        stage=stage,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
        bm=frontier_bm[improved],
        bn=frontier_bn[improved],
        bk=frontier_bk[improved],
        loop_orders=tuple(
            loop_order
            for keep, loop_order in zip(improved, frontier_loop_order_list)
            if keep
        ),
    )


def fused_stage_time(
    frontier: FusedTrafficFrontier,
    s_total: np.ndarray,
    tensor_roof: np.ndarray,
    cuda_roof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-area-node fused-stage time and HBM traffic, minimized over Pareto points.

    For each fused frontier point ``i`` (working set ``W_i = buffer_bytes`` incl.
    epilogue aux, HBM traffic ``T_i = hbm_bytes``), the optimal pipeline depth is
    ``C_best = min(floor(S_total / W_i), ceil(bw * latency / W_i))`` and
    ``BW_eff = min(bw, C_best * W_i / latency)``.  The stage time is
    ``count * max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T_i / BW_eff)``.
    """
    stage = frontier.stage
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    n = len(s_total)
    time_best = np.full(n, np.inf, dtype=float)
    traffic_best = np.full(n, np.nan, dtype=float)
    tensor_time = np.full(n, np.inf, dtype=float)
    cuda_time = np.full(n, np.inf, dtype=float)
    np.divide(stage.tensor_operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    np.divide(stage.cuda_operations, cuda_roof, out=cuda_time, where=cuda_roof > 0)
    for i in range(len(frontier.buffer_bytes)):
        w_i = float(frontier.buffer_bytes[i])
        t_i = float(frontier.traffic_bytes[i])
        c_max = np.floor(s_total / w_i)
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * latency_seconds / w_i))
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w_i / latency_seconds)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem_time = t_i / bw_eff
        time_i = stage.count * np.maximum.reduce([tensor_time, cuda_time, mem_time])
        time_i = np.where(valid, time_i, np.inf)
        better = time_i < time_best
        time_best = np.where(better, time_i, time_best)
        traffic_best = np.where(better, t_i, traffic_best)
    return time_best, traffic_best


def standard_stage_time(
    frontier: StandardTrafficFrontier,
    s_total: np.ndarray,
    tensor_roof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-area-node standard-GEMM stage time and traffic, minimized over Pareto
    points.  ``time = count * max(ops/tensor_roof, T_i / BW_eff)`` with
    ``BW_eff = min(bw, C_best * W_i / latency)``."""
    stage = frontier.stage
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    n = len(s_total)
    time_best = np.full(n, np.inf, dtype=float)
    traffic_best = np.full(n, np.nan, dtype=float)
    tensor_time = np.full(n, np.inf, dtype=float)
    np.divide(stage.operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    for i in range(len(frontier.buffer_bytes)):
        w_i = float(frontier.buffer_bytes[i])
        t_i = float(frontier.traffic_bytes[i])
        c_max = np.floor(s_total / w_i)
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * latency_seconds / w_i))
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w_i / latency_seconds)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem_time = t_i / bw_eff
        time_i = stage.count * np.maximum(tensor_time, mem_time)
        time_i = np.where(valid, time_i, np.inf)
        better = time_i < time_best
        time_best = np.where(better, time_i, time_best)
        traffic_best = np.where(better, t_i, traffic_best)
    return time_best, stage.count * traffic_best


def select_mapping_from_frontier(
    frontier: FusedTrafficFrontier | StandardTrafficFrontier,
    s_total: float,
    tensor_roof: float,
    cuda_roof: float | None = None,
) -> dict[str, object] | None:
    """Winning tiling + num_stages at a single (fixed) SMEM budget."""
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    stage = frontier.stage
    is_fused = isinstance(frontier, FusedTrafficFrontier)
    if is_fused:
        tensor_time = stage.tensor_operations / tensor_roof if tensor_roof > 0 else np.inf
        cuda_time = (
            stage.cuda_operations / cuda_roof
            if (cuda_roof is not None and cuda_roof > 0)
            else 0.0
        )
        ops_for_oi = stage.tensor_operations + stage.cuda_operations
    else:
        tensor_time = stage.operations / tensor_roof if tensor_roof > 0 else np.inf
        cuda_time = 0.0
        ops_for_oi = stage.operations
    best: dict[str, object] | None = None
    for i in range(len(frontier.buffer_bytes)):
        w_i = int(frontier.buffer_bytes[i])
        t_i = int(frontier.traffic_bytes[i])
        c_max = int(s_total // w_i)
        if c_max < 1:
            continue
        c_sat = int(np.ceil(bw * latency_seconds / w_i))
        c_best = min(c_max, c_sat)
        bw_eff = min(bw, c_best * w_i / latency_seconds)
        mem_time = t_i / bw_eff
        if is_fused:
            time_i = stage.count * max(tensor_time, cuda_time, mem_time)
        else:
            time_i = stage.count * max(tensor_time, mem_time)
        if best is None or time_i < best["time"]:  # type: ignore[typeddict-item]
            best = {
                "bm": int(frontier.bm[i]),
                "bn": int(frontier.bn[i]),
                "bk": int(frontier.bk[i]),
                "loop_order": frontier.loop_orders[i],
                "num_stages": c_best,
                "max_feasible_stages": c_max,
                "one_stage_smem": w_i,
                "traffic": t_i,
                "oi": ops_for_oi / t_i,
                "bw_eff": bw_eff,
                "time": time_i,
            }
    return best


def format_selected_mapping(
    frontier: FusedTrafficFrontier | StandardTrafficFrontier,
    s_total: float,
    tensor_roof: float,
    cuda_roof: float | None = None,
) -> str:
    mapping = select_mapping_from_frontier(frontier, s_total, tensor_roof, cuda_roof)
    if mapping is None:
        return "no mapping fits selected SMEM capacity"
    return (
        f"BM={mapping['bm']}, BN={mapping['bn']}, BK={mapping['bk']}, "
        f"loop_order={'-'.join(mapping['loop_order'])}, "
        f"num_stages={mapping['num_stages']} "
        f"(max_feasible={mapping['max_feasible_stages']}), "
        f"one_stage_smem={mapping['one_stage_smem'] / 2**20:.6f} MiB "
        f"({mapping['one_stage_smem']} bytes), "
        f"traffic={mapping['traffic'] / 2**20:.3f} MiB, "
        f"OI={mapping['oi']:.6f} FLOP/byte, "
        f"BW_eff={mapping['bw_eff'] / 1e12:.6f} TB/s"
    )


def reduction_time(task: ReductionTask, cuda_roof: np.ndarray) -> np.ndarray:
    memory_roof = task.operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(task.operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def vector_time(task: VectorTask, cuda_roof: np.ndarray) -> np.ndarray:
    memory_roof = task.operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(task.operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def streaming_cuda_time(
    operations: int | float, traffic_bytes: int | float, cuda_roof: np.ndarray
) -> np.ndarray:
    operational_intensity = operations / traffic_bytes
    memory_roof = operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def total_hbm_traffic_bytes(stage_traffic: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros_like(next(iter(stage_traffic.values())), dtype=float)
    for traffic_bytes in stage_traffic.values():
        total = total + traffic_bytes
    if INCLUDE_RMSNORM:
        total = total + RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes
    total = total + EXPERT_WEIGHTED_SUM_TASK.traffic_bytes
    total = total + RESIDUAL_ADD_TASK.traffic_bytes
    return total


def make_area_grid(step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.arange(step, 1.0, step)
    rc_values = []
    rt_values = []
    smem_values = []

    for rc in values:
        for rt in values:
            smem = 1.0 - rc - rt
            if smem > 0.0:
                rc_values.append(rc)
                rt_values.append(rt)
                smem_values.append(smem)

    return (
        np.array(rc_values, dtype=float),
        np.array(rt_values, dtype=float),
        np.array(smem_values, dtype=float),
    )


def write_csv(
    path: str,
    rc: np.ndarray,
    rt: np.ndarray,
    r_smem: np.ndarray,
    smem_bytes: np.ndarray,
    cuda_cores: np.ndarray,
    tensor_cores: np.ndarray,
    stage_times: dict[str, np.ndarray],
    stage_traffic: dict[str, np.ndarray],
    rmsnorm_time: np.ndarray,
    expert_weighted_sum_time: np.ndarray,
    residual_add_time: np.ndarray,
    total_time: np.ndarray,
    modeled_operations: float,
) -> None:
    with open(path, "w", newline="") as csvfile:
        fieldnames = [
            "rc",
            "rt",
            "r_smem",
            "smem_mib",
            "cuda_cores",
            "tensor_cores",
            "total_time_ms",
            "total_hbm_mib",
            "effective_tflops",
            "rmsnorm_square_reduction_time_ms",
            "expert_weighted_sum_time_ms",
            "residual_add_time_ms",
            *[f"{name}_time_ms" for name in stage_times],
            *[f"{name}_hbm_mib" for name in stage_traffic],
            "rmsnorm_square_reduction_hbm_mib",
            "expert_weighted_sum_hbm_mib",
            "residual_add_hbm_mib",
            *[f"{name}_oi_flops_per_byte" for name in stage_traffic],
            "expert_weighted_sum_oi_flops_per_byte",
            "residual_add_oi_flops_per_byte",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        total_hbm_traffic = total_hbm_traffic_bytes(stage_traffic)

        for index in range(len(rc)):
            row = {
                "rc": rc[index],
                "rt": rt[index],
                "r_smem": r_smem[index],
                "smem_mib": smem_bytes[index] / 2**20,
                "cuda_cores": cuda_cores[index],
                "tensor_cores": tensor_cores[index],
                "total_time_ms": total_time[index] * 1e3,
                "total_hbm_mib": total_hbm_traffic[index] / 2**20,
                "effective_tflops": modeled_operations / total_time[index] / 1e12,
                "rmsnorm_square_reduction_time_ms": rmsnorm_time[index] * 1e3,
                "expert_weighted_sum_time_ms": expert_weighted_sum_time[index] * 1e3,
                "residual_add_time_ms": residual_add_time[index] * 1e3,
            }
            for name, time_seconds in stage_times.items():
                row[f"{name}_time_ms"] = time_seconds[index] * 1e3
            for name, traffic_bytes in stage_traffic.items():
                row[f"{name}_hbm_mib"] = traffic_bytes[index] / 2**20
            row["rmsnorm_square_reduction_hbm_mib"] = (
                RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20
            )
            row["expert_weighted_sum_hbm_mib"] = (
                EXPERT_WEIGHTED_SUM_TASK.traffic_bytes / 2**20
            )
            row["residual_add_hbm_mib"] = RESIDUAL_ADD_TASK.traffic_bytes / 2**20
            for name, traffic_bytes in stage_traffic.items():
                stage_ops = _stage_ops_by_name[name]
                row[f"{name}_oi_flops_per_byte"] = stage_ops / traffic_bytes[index]
            row["expert_weighted_sum_oi_flops_per_byte"] = (
                EXPERT_WEIGHTED_SUM_TASK.operational_intensity
            )
            row["residual_add_oi_flops_per_byte"] = (
                RESIDUAL_ADD_TASK.operational_intensity
            )
            writer.writerow(row)


def plot_results(
    rc: np.ndarray,
    rt: np.ndarray,
    total_time: np.ndarray,
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Path("result").mkdir(exist_ok=True)
    valid = np.isfinite(total_time) & (total_time > 0)
    total_time_ms = total_time[valid] * 1e3

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        rt[valid],
        rc[valid],
        c=total_time_ms,
        s=8,
        cmap="viridis_r",
        norm=LogNorm(vmin=total_time_ms.min(), vmax=total_time_ms.max()),
    )
    plt.colorbar(scatter, label="Total time (ms)")
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title("Fused FFN Decode Time Across CUDA/Tensor/SMEM Area Split")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


_stage_ops_by_name: dict[str, float] = {}


def main() -> None:
    rc, rt, r_smem = make_area_grid(AREA_GRID_STEP)
    smem_bytes = r_smem * A_total / A_bit / 8
    cuda_cores = np.floor(rc * A_total / A_cuda_core)
    tensor_cores = np.floor(rt * A_total / A_tensor_core)
    cuda_roof = cuda_cores * ACTIVATION_FLOPS_PER_CUDA_CORE
    tensor_roof = tensor_cores * TENSOR_FLOPS
    flops_per_cuda_core_cycle = ACTIVATION_FLOPS_PER_CUDA_CORE / CUDA_CLOCK_HZ

    _stage_ops_by_name.clear()
    distribution = (
        expert_token_distribution() if USE_RANDOM_EXPERT_DISTRIBUTION else None
    )
    if distribution is None:
        fused_stages = make_fused_stages()
        standard_stages = make_standard_stages()
        stage_weights = {
            stage.name: 1.0 for stage in [*fused_stages, *standard_stages]
        }
        aggregate_names = {
            stage.name: stage.name for stage in [*fused_stages, *standard_stages]
        }
    else:
        (
            fused_stages,
            standard_stages,
            stage_weights,
            aggregate_names,
        ) = make_random_expert_stages(distribution)

    fused_frontiers = [build_fused_frontier(stage) for stage in fused_stages]
    standard_frontiers = [
        build_standard_frontier(stage) for stage in standard_stages
    ]

    stage_times: dict[str, np.ndarray] = {}
    stage_traffic: dict[str, np.ndarray] = {}
    for frontier in fused_frontiers:
        aggregate_name = aggregate_names[frontier.stage.name]
        weight = stage_weights[frontier.stage.name]
        _stage_ops_by_name[aggregate_name] = (
            _stage_ops_by_name.get(aggregate_name, 0.0)
            + weight * frontier.stage.total_operations
        )
        time_seconds, traffic_bytes = fused_stage_time(
            frontier,
            smem_bytes,
            tensor_roof,
            cuda_roof,
        )
        weighted_time = weight * time_seconds
        weighted_traffic = weight * frontier.stage.count * traffic_bytes
        if aggregate_name in stage_times:
            stage_times[aggregate_name] = stage_times[aggregate_name] + weighted_time
            stage_traffic[aggregate_name] = (
                stage_traffic[aggregate_name] + weighted_traffic
            )
        else:
            stage_times[aggregate_name] = weighted_time
            stage_traffic[aggregate_name] = weighted_traffic
    for frontier in standard_frontiers:
        aggregate_name = aggregate_names[frontier.stage.name]
        weight = stage_weights[frontier.stage.name]
        _stage_ops_by_name[aggregate_name] = (
            _stage_ops_by_name.get(aggregate_name, 0.0)
            + weight * frontier.stage.total_operations
        )
        time_seconds, traffic_bytes = standard_stage_time(
            frontier,
            smem_bytes,
            tensor_roof,
        )
        weighted_time = weight * time_seconds
        weighted_traffic = weight * traffic_bytes
        if aggregate_name in stage_times:
            stage_times[aggregate_name] = stage_times[aggregate_name] + weighted_time
            stage_traffic[aggregate_name] = (
                stage_traffic[aggregate_name] + weighted_traffic
            )
        else:
            stage_times[aggregate_name] = weighted_time
            stage_traffic[aggregate_name] = weighted_traffic

    rmsnorm_time = (
        reduction_time(RMSNORM_SQUARE_REDUCTION_TASK, cuda_roof)
        if INCLUDE_RMSNORM
        else np.zeros(len(rc), dtype=float)
    )
    expert_weighted_sum_time = streaming_cuda_time(
        EXPERT_WEIGHTED_SUM_TASK.operations,
        EXPERT_WEIGHTED_SUM_TASK.traffic_bytes,
        cuda_roof,
    )
    residual_add_time = vector_time(RESIDUAL_ADD_TASK, cuda_roof)
    total_time = np.sum(
        np.array(
            [
                *stage_times.values(),
                rmsnorm_time,
                expert_weighted_sum_time,
                residual_add_time,
            ]
        ),
        axis=0,
    )
    modeled_operations = (
        sum(_stage_ops_by_name.values())
        + (RMSNORM_SQUARE_REDUCTION_TASK.operations if INCLUDE_RMSNORM else 0)
        + EXPERT_WEIGHTED_SUM_TASK.operations
        + RESIDUAL_ADD_TASK.operations
    )

    Path("result").mkdir(exist_ok=True)
    csv_path, plot_path = output_paths()
    write_csv(
        csv_path,
        rc,
        rt,
        r_smem,
        smem_bytes,
        cuda_cores,
        tensor_cores,
        stage_times,
        stage_traffic,
        rmsnorm_time,
        expert_weighted_sum_time,
        residual_add_time,
        total_time,
        modeled_operations,
    )
    plot_results(rc, rt, total_time, plot_path)

    best_index = int(np.nanargmin(total_time))
    effective_flops = modeled_operations / total_time[best_index]
    total_hbm_traffic = total_hbm_traffic_bytes(stage_traffic)

    print("\n=== Configuration ===")
    print(f"Fused stages: {len(fused_stages)}")
    print(f"Standard GEMM stages: {len(standard_stages)}")
    print(
        "Fused traffic model: "
        + (
            "register-accumulator loop orders only"
            if USE_REGISTER_ACCUMULATOR_MAPPINGS
            else "original fused all-loop-order mapspace"
        )
    )
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        allowed = ["-".join(loop_order) for loop_order in REGISTER_ACCUMULATOR_LOOP_ORDERS]
        print(f"Fused allowed loop orders: {', '.join(allowed)}")
    print(f"Output CSV: {csv_path}")
    print(f"HBM latency: {HBM_LATENCY_CYCLES} cycles")
    print(f"Batch tokens: {BATCH_TOKENS}")
    print(f"Router top-k: {ROUTER_TOP_K}")
    print(f"Expert token split: {EVEN_SPLIT.summary()}")
    if EVEN_SPLIT.ceil_tokens < TENSOR_CORE_MIN_BM:
        print(
            f"  per-expert GEMM M padded to {TENSOR_CORE_MIN_BM} "
            f"(tensor core underutilized; real tokens/expert < {TENSOR_CORE_MIN_BM})"
        )
    print(
        "Expert distribution model: "
        + (
            "random binomial expected value"
            if USE_RANDOM_EXPERT_DISTRIBUTION
            else "even deterministic tokens per expert"
        )
    )
    if distribution is not None:
        support_counts = [tokens for tokens, _ in distribution.support]
        print(
            "Per-expert token count: "
            f"Binomial(n={distribution.batch_tokens}, "
            f"p={distribution.selection_probability:.8f})"
        )
        print(
            "Per-expert token mean/stddev: "
            f"{distribution.mean:.6f}/"
            f"{np.sqrt(distribution.variance):.6f}"
        )
        print(
            "Retained token-count support: "
            f"{min(support_counts)}..{max(support_counts)} "
            f"({len(support_counts)} counts), "
            f"probability mass {distribution.retained_probability_mass:.12f}"
        )
    print(f"CUDA FLOP/cycle/core: {flops_per_cuda_core_cycle:.6f}")
    print(f"RMSNorm square-reduction enabled: {INCLUDE_RMSNORM}")
    print(
        "RMSNorm square-reduction OI: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity:.6f} FLOP/byte"
    )

    print("\n=== Best Area Point ===")
    print(f"Best rc: {rc[best_index]:.6g}")
    print(f"Best rt: {rt[best_index]:.6g}")
    print(f"Best SMEM fraction: {r_smem[best_index]:.6g}")
    print(f"SMEM: {smem_bytes[best_index] / 2**20:.3f} MiB")
    print(f"CUDA Cores: {int(cuda_cores[best_index])}")
    print(f"Tensor Cores: {int(tensor_cores[best_index])}")
    print(f"Total execution time: {total_time[best_index] * 1e3:.6f} ms")
    print(f"Total HBM traffic: {total_hbm_traffic[best_index] / 2**20:.3f} MiB")
    print(f"Effective throughput: {effective_flops / 1e12:.3f} TFLOP/s")

    print("\n=== Vector / Reduction Stages ===")
    print(f"rmsnorm_square_reduction time: {rmsnorm_time[best_index] * 1e3:.6f} ms")
    print(
        "rmsnorm_square_reduction HBM traffic: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20:.3f} MiB"
    )

    print("\n=== GEMM / Fused GEMM Stages ===")
    frontiers_by_aggregate: dict[
        str, list[FusedTrafficFrontier | StandardTrafficFrontier]
    ] = {}
    for frontier in [*fused_frontiers, *standard_frontiers]:
        aggregate_name = aggregate_names[frontier.stage.name]
        frontiers_by_aggregate.setdefault(aggregate_name, []).append(frontier)
    for name in stage_times:
        traffic = stage_traffic[name][best_index]
        print(f"\n{name}")
        print(f"  time: {stage_times[name][best_index] * 1e3:.6f} ms")
        print(f"  HBM traffic: {traffic / 2**20:.3f} MiB")
        print(f"  OI: {_stage_ops_by_name[name] / traffic:.6f} FLOP/byte")
        stage_frontiers = frontiers_by_aggregate.get(name, [])
        if len(stage_frontiers) == 1:
            print(
                "  mapping: "
                f"{format_selected_mapping(stage_frontiers[0], smem_bytes[best_index], tensor_roof[best_index], cuda_roof[best_index])}"
            )
        elif stage_frontiers:
            print("  constituent mappings:")
            for stage_frontier in stage_frontiers:
                weight = stage_weights[stage_frontier.stage.name]
                print(
                    f"    {stage_frontier.stage.name} "
                    f"(expected_count={weight:.8g}): "
                    f"{format_selected_mapping(stage_frontier, smem_bytes[best_index], tensor_roof[best_index], cuda_roof[best_index])}"
                )

    print("\n=== Remaining Vector Stages ===")
    print(
        "expert_weighted_sum time: "
        f"{expert_weighted_sum_time[best_index] * 1e3:.6f} ms"
    )
    print(
        "expert_weighted_sum HBM traffic: "
        f"{EXPERT_WEIGHTED_SUM_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    print(f"residual_add time: {residual_add_time[best_index] * 1e3:.6f} ms")
    print(f"residual_add HBM traffic: {RESIDUAL_ADD_TASK.traffic_bytes / 2**20:.3f} MiB")


if __name__ == "__main__":
    main()
