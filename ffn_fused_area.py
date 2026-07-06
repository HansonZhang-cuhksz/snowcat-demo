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
BATCH_TOKENS = 8192
EXPERTS = 512
TOKENS_PER_EXPERT = 128
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
ROUTER_TOP_K = EXPERTS * TOKENS_PER_EXPERT // BATCH_TOKENS

SWIGLU_FLOPS_PER_ELEMENT = 8.0
INCLUDE_RMSNORM = True
AREA_GRID_STEP = 0.001

# False preserves the original fused Snowcat-style mapspace and output
# filenames.  True restricts fused GEMM mappings to M-N-K and N-M-K, where the
# output accumulator tile stays live through the K reduction.
USE_REGISTER_ACCUMULATOR_MAPPINGS = False


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

    router = FusedGemmStage(
        name="router_rms_scale",
        m=BATCH_TOKENS,
        n=EXPERTS,
        k=HIDDEN_SIZE,
        count=1,
        tensor_operations=2 * BATCH_TOKENS * EXPERTS * HIDDEN_SIZE,
        cuda_operations=BATCH_TOKENS * EXPERTS,
        traffic_points_fn=lambda: router_traffic_points_fn(
            BATCH_TOKENS,
            EXPERTS,
            HIDDEN_SIZE,
            BYTE_PER_ELEMENT,
        ),
    )

    up_gate = FusedGemmStage(
        name="up_gate_rms_swiglu_x512",
        m=TOKENS_PER_EXPERT,
        n=2 * INTERMEDIATE_SIZE,
        k=HIDDEN_SIZE,
        count=EXPERTS,
        tensor_operations=(
            2 * TOKENS_PER_EXPERT * (2 * INTERMEDIATE_SIZE) * HIDDEN_SIZE
        ),
        cuda_operations=(
            TOKENS_PER_EXPERT * (2 * INTERMEDIATE_SIZE)
            + TOKENS_PER_EXPERT * INTERMEDIATE_SIZE * SWIGLU_FLOPS_PER_ELEMENT
        ),
        traffic_points_fn=lambda: up_gate_traffic_points_fn(
            TOKENS_PER_EXPERT,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            BYTE_PER_ELEMENT,
        ),
    )

    return [router, up_gate]


def make_standard_stages() -> list[StandardGemmStage]:
    down = StandardGemmStage(
        name="down_x512",
        m=TOKENS_PER_EXPERT,
        n=HIDDEN_SIZE,
        k=INTERMEDIATE_SIZE,
        count=EXPERTS,
    )
    return [down]


def output_paths() -> tuple[str, str]:
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        return (
            "./result/ffn_fused_area_register_accumulator_times.csv",
            "./result/ffn_fused_area_register_accumulator_total_time.png",
        )
    return (
        "./result/ffn_fused_area_times.csv",
        "./result/ffn_fused_area_total_time.png",
    )


def build_fused_frontier(stage: FusedGemmStage) -> FusedTrafficFrontier:
    pairs = sorted(
        (point.buffer_bytes, point.hbm_bytes) for point in stage.traffic_points_fn()
    )
    buffers = np.fromiter((buffer for buffer, _ in pairs), dtype=np.int64)
    traffic = np.fromiter((bytes_ for _, bytes_ in pairs), dtype=np.int64)

    best_traffic = np.minimum.accumulate(traffic)
    last_at_buffer = np.r_[buffers[1:] != buffers[:-1], True]
    frontier_buffers = buffers[last_at_buffer]
    frontier_traffic = best_traffic[last_at_buffer]
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return FusedTrafficFrontier(
        stage=stage,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
    )


def build_standard_frontier(stage: StandardGemmStage) -> StandardTrafficFrontier:
    workload = GemmWorkload(
        m=stage.m,
        k=stage.k,
        n=stage.n,
        bytes_per_element=BYTE_PER_ELEMENT,
    )
    pairs = sorted(
        (point.buffer_bytes, point.backing_store_bytes)
        for point in enumerate_mappings(workload)
    )
    buffers = np.fromiter((buffer for buffer, _ in pairs), dtype=np.int64)
    traffic = np.fromiter((bytes_ for _, bytes_ in pairs), dtype=np.int64)

    best_traffic = np.minimum.accumulate(traffic)
    last_at_buffer = np.r_[buffers[1:] != buffers[:-1], True]
    frontier_buffers = buffers[last_at_buffer]
    frontier_traffic = best_traffic[last_at_buffer]
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return StandardTrafficFrontier(
        stage=stage,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
    )


def min_traffic_from_frontier(
    frontier: FusedTrafficFrontier | StandardTrafficFrontier, capacities: np.ndarray
) -> np.ndarray:
    indexes = np.searchsorted(frontier.buffer_bytes, capacities, side="right") - 1
    min_traffic = np.full(len(capacities), np.nan, dtype=float)
    valid = indexes >= 0
    min_traffic[valid] = frontier.traffic_bytes[indexes[valid]]
    return min_traffic


def fused_stage_time(
    frontier: FusedTrafficFrontier,
    capacities: np.ndarray,
    tensor_roof: np.ndarray,
    cuda_roof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stage = frontier.stage
    min_traffic = min_traffic_from_frontier(frontier, capacities)

    tensor_time = np.full(len(capacities), np.nan, dtype=float)
    cuda_time = np.full(len(capacities), np.nan, dtype=float)
    memory_time = min_traffic / bw

    np.divide(
        stage.tensor_operations,
        tensor_roof,
        out=tensor_time,
        where=tensor_roof > 0,
    )
    np.divide(
        stage.cuda_operations,
        cuda_roof,
        out=cuda_time,
        where=cuda_roof > 0,
    )

    per_invocation = np.maximum.reduce([tensor_time, cuda_time, memory_time])
    return stage.count * per_invocation, min_traffic


def standard_stage_time(
    frontier: StandardTrafficFrontier,
    capacities: np.ndarray,
    tensor_roof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stage = frontier.stage
    min_traffic = min_traffic_from_frontier(frontier, capacities)
    oi = stage.operations / min_traffic
    memory_roof = oi * bw
    peak = np.minimum(memory_roof, tensor_roof)

    time_seconds = np.full(len(capacities), np.nan, dtype=float)
    np.divide(stage.operations, peak, out=time_seconds, where=peak > 0)
    return stage.count * time_seconds, stage.count * min_traffic


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

        for index in range(len(rc)):
            row = {
                "rc": rc[index],
                "rt": rt[index],
                "r_smem": r_smem[index],
                "smem_mib": smem_bytes[index] / 2**20,
                "cuda_cores": cuda_cores[index],
                "tensor_cores": tensor_cores[index],
                "total_time_ms": total_time[index] * 1e3,
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
    if ROUTER_TOP_K * BATCH_TOKENS != EXPERTS * TOKENS_PER_EXPERT:
        raise ValueError("Expert token count must equal batch tokens times top-k")

    rc, rt, r_smem = make_area_grid(AREA_GRID_STEP)
    smem_bytes = r_smem * A_total / A_bit / 8
    cuda_cores = np.floor(rc * A_total / A_cuda_core)
    tensor_cores = np.floor(rt * A_total / A_tensor_core)
    cuda_roof = cuda_cores * ACTIVATION_FLOPS_PER_CUDA_CORE
    tensor_roof = tensor_cores * TENSOR_FLOPS
    flops_per_cuda_core_cycle = ACTIVATION_FLOPS_PER_CUDA_CORE / CUDA_CLOCK_HZ

    fused_stages = make_fused_stages()
    standard_stages = make_standard_stages()
    _stage_ops_by_name.update(
        {stage.name: stage.total_operations for stage in fused_stages}
    )
    _stage_ops_by_name.update(
        {stage.name: stage.total_operations for stage in standard_stages}
    )
    fused_frontiers = [build_fused_frontier(stage) for stage in fused_stages]
    standard_frontiers = [
        build_standard_frontier(stage) for stage in standard_stages
    ]

    stage_times: dict[str, np.ndarray] = {}
    stage_traffic: dict[str, np.ndarray] = {}
    for frontier in fused_frontiers:
        time_seconds, traffic_bytes = fused_stage_time(
            frontier,
            smem_bytes,
            tensor_roof,
            cuda_roof,
        )
        stage_times[frontier.stage.name] = time_seconds
        stage_traffic[frontier.stage.name] = frontier.stage.count * traffic_bytes
    for frontier in standard_frontiers:
        time_seconds, traffic_bytes = standard_stage_time(
            frontier,
            smem_bytes,
            tensor_roof,
        )
        stage_times[frontier.stage.name] = time_seconds
        stage_traffic[frontier.stage.name] = traffic_bytes

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
        sum(stage.total_operations for stage in fused_stages)
        + sum(stage.total_operations for stage in standard_stages)
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
    print(f"Router top-k implied by token routing: {ROUTER_TOP_K}")
    print(f"CUDA FLOP/cycle/core: {flops_per_cuda_core_cycle:.6f}")
    print(f"RMSNorm square-reduction enabled: {INCLUDE_RMSNORM}")
    print(
        "RMSNorm square-reduction OI: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Best rc: {rc[best_index]:.6g}")
    print(f"Best rt: {rt[best_index]:.6g}")
    print(f"Best SMEM fraction: {r_smem[best_index]:.6g}")
    print(f"SMEM: {smem_bytes[best_index] / 2**20:.3f} MiB")
    print(f"CUDA Cores: {int(cuda_cores[best_index])}")
    print(f"Tensor Cores: {int(tensor_cores[best_index])}")
    print(f"Total execution time: {total_time[best_index] * 1e3:.6f} ms")
    print(f"Effective throughput: {effective_flops / 1e12:.3f} TFLOP/s")
    print(f"rmsnorm_square_reduction time: {rmsnorm_time[best_index] * 1e3:.6f} ms")
    print(
        "rmsnorm_square_reduction HBM traffic: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    for stage in [*fused_stages, *standard_stages]:
        name = stage.name
        traffic = stage_traffic[name][best_index]
        print(f"{name} time: {stage_times[name][best_index] * 1e3:.6f} ms")
        print(f"{name} HBM traffic: {traffic / 2**20:.3f} MiB")
        print(f"{name} OI: {stage.total_operations / traffic:.6f} FLOP/byte")
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
