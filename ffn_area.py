from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from register_accumulator_traffic import (
    FULLY_TILED_REGISTER_ACCUMULATOR_LOOP_ORDERS,
    enumerate_register_accumulator_mappings,
)
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


# Chip constants
# A_total = 694.116 * 10**6              # um^2
A_total = 136.29 * 10**6              # um^2
A_bit = 0.0864                         # um^2/bit
logic_density = 39.98                  # MTr/mm^2 == transistors/um^2
bw = 2.04 * 10**12                     # byte/s

num_sm = 1
A_total = A_total / num_sm
bw = bw / num_sm

# Architecture placeholders. Replace these with measured/spec-sheet values.
CUDA_CORE_TRANSISTORS = 0.2 * 10**6     # transistors/CUDA core placeholder
TENSOR_CORE_TRANSISTORS = 6.0 * 10**6   # transistors/tensor core placeholder
TENSOR_FLOPS = 512 * 1.00 * 10**9       # flops/s/tensor core placeholder
CUDA_CLOCK_HZ = 1410 * 10**6            # Hz
ACTIVATION_FLOPS_PER_CUDA_CORE = 5.64 * 10 ** 9      # flops/s/CUDA core placeholder

A_cuda_core = CUDA_CORE_TRANSISTORS / logic_density
A_tensor_core = TENSOR_CORE_TRANSISTORS / logic_density

# FFN workload assumptions
BYTE_PER_ELEMENT = 2
BATCH_TOKENS = 8192
EXPERTS = 512
TOKENS_PER_EXPERT = 128
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
ROUTER_TOP_K = EXPERTS * TOKENS_PER_EXPERT // BATCH_TOKENS

# Fused SwiGLU activation: SiLU(gate) * up. FLOP accounting for exp/sigmoid is
# implementation dependent, so keep this as a measured/estimated placeholder.
SWIGLU_FLOPS_PER_ELEMENT = 8.0
INCLUDE_RMSNORM = True

CPU_WORKERS = 32
PARALLEL_FRONTIER_MIN_GROUPS = CPU_WORKERS
AREA_GRID_STEP = 0.001

# False preserves the original Snowcat mapspace and output filenames.  True
# restricts GEMM mappings to loop orders that keep output accumulators live
# through their K reduction, closer to real tensor-core GEMM schedules.
USE_REGISTER_ACCUMULATOR_MAPPINGS = True


@dataclass(frozen=True)
class GemmTask:
    name: str
    m: int
    n: int
    k: int

    @property
    def operations(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass(frozen=True)
class GemmTaskGroup:
    label: str
    task: GemmTask
    count: int

    @property
    def operations(self) -> int:
        return self.count * self.task.operations


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
class ReductionTask:
    name: str
    rows: int
    columns: int
    bytes_per_input: int
    bytes_per_output: int

    @property
    def operations(self) -> int:
        square_ops = self.rows * self.columns
        reduction_adds = self.rows * (self.columns - 1)
        return square_ops + reduction_adds

    @property
    def traffic_bytes(self) -> int:
        input_reads = self.rows * self.columns * self.bytes_per_input
        output_writes = self.rows * self.bytes_per_output
        return input_reads + output_writes

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
class TrafficFrontier:
    label: str
    count: int
    operations: int
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray


GEMM_TASKS = [
    GemmTask("router", BATCH_TOKENS, EXPERTS, HIDDEN_SIZE),
    GemmTask("up_gate", TOKENS_PER_EXPERT, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
    GemmTask("down", TOKENS_PER_EXPERT, HIDDEN_SIZE, INTERMEDIATE_SIZE),
]

ACTIVATION_TASK = VectorTask(
    name="activation",
    elements=TOKENS_PER_EXPERT * INTERMEDIATE_SIZE,
    count=EXPERTS,
    flops_per_element=SWIGLU_FLOPS_PER_ELEMENT,
    bytes_per_element_traffic=3 * BYTE_PER_ELEMENT,
)

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


def group_gemm_tasks(tasks: list[GemmTask]) -> list[GemmTaskGroup]:
    groups: dict[tuple[str, int, int, int], tuple[GemmTask, int]] = {}
    for task in tasks:
        key = (task.name, task.m, task.n, task.k)
        if key in groups:
            original_task, count = groups[key]
            groups[key] = (original_task, count + 1)
        else:
            groups[key] = (task, 1)

    task_groups = []
    for task, count in groups.values():
        if task.name in {"up_gate", "down"}:
            count *= EXPERTS
        label = task.name if count == 1 else f"{task.name}_x{count}"
        task_groups.append(GemmTaskGroup(label=label, task=task, count=count))
    return task_groups


def build_traffic_frontier(group: GemmTaskGroup) -> TrafficFrontier:
    task = group.task
    workload = GemmWorkload(
        m=task.m,
        k=task.k,
        n=task.n,
        bytes_per_element=BYTE_PER_ELEMENT,
    )
    mapping_points = (
        enumerate_register_accumulator_mappings(workload)
        if USE_REGISTER_ACCUMULATOR_MAPPINGS
        else enumerate_mappings(workload)
    )
    pairs = sorted(
        (point.buffer_bytes, point.backing_store_bytes)
        for point in mapping_points
    )
    buffers = np.fromiter((buffer for buffer, _ in pairs), dtype=np.int64)
    traffic = np.fromiter((bytes_ for _, bytes_ in pairs), dtype=np.int64)

    best_traffic = np.minimum.accumulate(traffic)
    last_at_buffer = np.r_[buffers[1:] != buffers[:-1], True]
    frontier_buffers = buffers[last_at_buffer]
    frontier_traffic = best_traffic[last_at_buffer]
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return TrafficFrontier(
        label=group.label,
        count=group.count,
        operations=group.task.operations,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
    )


def output_paths() -> tuple[str, str, str]:
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        return (
            "./result/ffn_area_register_accumulator_times.csv",
            "./result/ffn_area_register_accumulator_total_time.png",
            "./result/ffn_area_register_accumulator_activation_time.png",
        )
    return (
        "./result/ffn_area_times.csv",
        "./result/ffn_area_total_time.png",
        "./result/ffn_area_activation_time.png",
    )


def build_frontiers(task_groups: list[GemmTaskGroup]) -> tuple[list[TrafficFrontier], str]:
    if len(task_groups) < PARALLEL_FRONTIER_MIN_GROUPS:
        return [build_traffic_frontier(group) for group in task_groups], "serial"

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
        frontiers = list(
            executor.map(build_traffic_frontier, task_groups, chunksize=1)
        )
    return frontiers, f"{CPU_WORKERS}-process"


def min_traffic_from_frontier(
    frontier: TrafficFrontier, capacities: np.ndarray
) -> np.ndarray:
    indexes = np.searchsorted(frontier.buffer_bytes, capacities, side="right") - 1
    min_traffic = np.full(len(capacities), np.nan, dtype=float)
    valid = indexes >= 0
    min_traffic[valid] = frontier.traffic_bytes[indexes[valid]]
    return min_traffic


def gemm_time_from_frontier(
    frontier: TrafficFrontier, capacities: np.ndarray, tensor_roof: np.ndarray
) -> np.ndarray:
    min_traffic = min_traffic_from_frontier(frontier, capacities)
    oi = frontier.operations / min_traffic
    memory_roof = oi * bw
    peak = np.minimum(memory_roof, tensor_roof)

    time_seconds = np.full(len(capacities), np.nan, dtype=float)
    np.divide(frontier.operations, peak, out=time_seconds, where=peak > 0)
    return frontier.count * time_seconds


def vector_time(task: VectorTask, cuda_roof: np.ndarray) -> np.ndarray:
    memory_roof = task.operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(task.operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def reduction_time(task: ReductionTask, cuda_roof: np.ndarray) -> np.ndarray:
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
    task_times: dict[str, np.ndarray],
    task_traffic: dict[str, np.ndarray],
    rmsnorm_time: np.ndarray,
    activation_time: np.ndarray,
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
            "activation_oi_flops_per_byte",
            "rmsnorm_square_reduction_oi_flops_per_byte",
            "expert_weighted_sum_oi_flops_per_byte",
            "residual_add_oi_flops_per_byte",
            *[f"{name}_time_ms" for name in task_times],
            "rmsnorm_square_reduction_time_ms",
            "activation_time_ms",
            "expert_weighted_sum_time_ms",
            "residual_add_time_ms",
            *[f"{name}_hbm_mib" for name in task_traffic],
            "rmsnorm_square_reduction_hbm_mib",
            "activation_hbm_mib",
            "expert_weighted_sum_hbm_mib",
            "residual_add_hbm_mib",
            *[f"{name}_oi_flops_per_byte" for name in task_traffic],
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
                "activation_oi_flops_per_byte": ACTIVATION_TASK.operational_intensity,
                "rmsnorm_square_reduction_oi_flops_per_byte": (
                    RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity
                ),
                "expert_weighted_sum_oi_flops_per_byte": (
                    EXPERT_WEIGHTED_SUM_TASK.operational_intensity
                ),
                "residual_add_oi_flops_per_byte": (
                    RESIDUAL_ADD_TASK.operational_intensity
                ),
            }
            for name, time_seconds in task_times.items():
                row[f"{name}_time_ms"] = time_seconds[index] * 1e3
            row["rmsnorm_square_reduction_time_ms"] = rmsnorm_time[index] * 1e3
            row["activation_time_ms"] = activation_time[index] * 1e3
            row["expert_weighted_sum_time_ms"] = (
                expert_weighted_sum_time[index] * 1e3
            )
            row["residual_add_time_ms"] = residual_add_time[index] * 1e3
            for name, traffic_bytes in task_traffic.items():
                row[f"{name}_hbm_mib"] = traffic_bytes[index] / 2**20
            row["rmsnorm_square_reduction_hbm_mib"] = (
                RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20
            )
            row["activation_hbm_mib"] = ACTIVATION_TASK.traffic_bytes / 2**20
            row["expert_weighted_sum_hbm_mib"] = (
                EXPERT_WEIGHTED_SUM_TASK.traffic_bytes / 2**20
            )
            row["residual_add_hbm_mib"] = RESIDUAL_ADD_TASK.traffic_bytes / 2**20
            for name, traffic_bytes in task_traffic.items():
                row[f"{name}_oi_flops_per_byte"] = (
                    task_operations_by_name[name] / traffic_bytes[index]
                )
            writer.writerow(row)


def plot_results(
    rc: np.ndarray,
    rt: np.ndarray,
    total_time: np.ndarray,
    activation_time: np.ndarray,
    total_time_path: str,
    activation_time_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Path("result").mkdir(exist_ok=True)
    valid = np.isfinite(total_time) & (total_time > 0) & (activation_time > 0)

    plt.figure(figsize=(10, 7))
    total_time_ms = total_time[valid] * 1e3
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
    plt.title("FFN Decode Time Across CUDA/Tensor/SMEM Area Split")
    plt.tight_layout()
    plt.savefig(total_time_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    activation_time_us = activation_time[valid] * 1e6
    scatter = plt.scatter(
        rt[valid],
        rc[valid],
        c=activation_time_us,
        s=8,
        cmap="magma_r",
        norm=LogNorm(vmin=activation_time_us.min(), vmax=activation_time_us.max()),
    )
    plt.colorbar(scatter, label="Activation time (us)")
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title("SwiGLU/SiLU Activation Time Across Area Split")
    plt.tight_layout()
    plt.savefig(activation_time_path, dpi=160)
    plt.close()


task_operations_by_name: dict[str, int] = {}


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

    task_groups = group_gemm_tasks(GEMM_TASKS)
    frontiers, frontier_mode = build_frontiers(task_groups)
    gemm_operations = sum(frontier.count * frontier.operations for frontier in frontiers)
    task_operations_by_name.update(
        {frontier.label: frontier.count * frontier.operations for frontier in frontiers}
    )
    rmsnorm_operations = (
        RMSNORM_SQUARE_REDUCTION_TASK.operations if INCLUDE_RMSNORM else 0
    )
    modeled_operations = (
        gemm_operations
        + ACTIVATION_TASK.operations
        + rmsnorm_operations
        + EXPERT_WEIGHTED_SUM_TASK.operations
        + RESIDUAL_ADD_TASK.operations
    )

    task_times = {
        frontier.label: gemm_time_from_frontier(frontier, smem_bytes, tensor_roof)
        for frontier in frontiers
    }
    task_traffic = {
        frontier.label: frontier.count
        * min_traffic_from_frontier(frontier, smem_bytes)
        for frontier in frontiers
    }
    rmsnorm_time = (
        reduction_time(RMSNORM_SQUARE_REDUCTION_TASK, cuda_roof)
        if INCLUDE_RMSNORM
        else np.zeros(len(rc), dtype=float)
    )
    activation_time = vector_time(ACTIVATION_TASK, cuda_roof)
    expert_weighted_sum_time = streaming_cuda_time(
        EXPERT_WEIGHTED_SUM_TASK.operations,
        EXPERT_WEIGHTED_SUM_TASK.traffic_bytes,
        cuda_roof,
    )
    residual_add_time = vector_time(RESIDUAL_ADD_TASK, cuda_roof)
    total_time = np.sum(
        np.array(
            [
                *task_times.values(),
                rmsnorm_time,
                activation_time,
                expert_weighted_sum_time,
                residual_add_time,
            ]
        ),
        axis=0,
    )

    Path("result").mkdir(exist_ok=True)
    csv_path, total_time_plot_path, activation_time_plot_path = output_paths()
    write_csv(
        csv_path,
        rc,
        rt,
        r_smem,
        smem_bytes,
        cuda_cores,
        tensor_cores,
        task_times,
        task_traffic,
        rmsnorm_time,
        activation_time,
        expert_weighted_sum_time,
        residual_add_time,
        total_time,
        modeled_operations,
    )
    plot_results(
        rc,
        rt,
        total_time,
        activation_time,
        total_time_plot_path,
        activation_time_plot_path,
    )

    best_index = int(np.nanargmin(total_time))
    effective_flops = modeled_operations / total_time[best_index]

    print(f"CPU workers available: {CPU_WORKERS}")
    print(f"Frontier build mode: {frontier_mode}")
    print(
        "Traffic model: "
        + (
            "register-accumulator loop orders only"
            if USE_REGISTER_ACCUMULATOR_MAPPINGS
            else "original Snowcat all-loop-order mapspace"
        )
    )
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        allowed = [
            "-".join(loop_order)
            for loop_order in FULLY_TILED_REGISTER_ACCUMULATOR_LOOP_ORDERS
        ]
        print(f"Fully tiled allowed loop orders: {', '.join(allowed)}")
    print(f"Output CSV: {csv_path}")
    print(f"Unique GEMM groups: {len(task_groups)}")
    print(f"Router top-k implied by token routing: {ROUTER_TOP_K}")
    print(f"CUDA FLOP/cycle/core: {flops_per_cuda_core_cycle:.6f}")
    print(f"RMSNorm square-reduction enabled: {INCLUDE_RMSNORM}")
    print(
        "RMSNorm square-reduction OI: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Activation OI: {ACTIVATION_TASK.operational_intensity:.6f} FLOP/byte")
    print(
        "Expert weighted-sum OI: "
        f"{EXPERT_WEIGHTED_SUM_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Residual add OI: {RESIDUAL_ADD_TASK.operational_intensity:.6f} FLOP/byte")
    print(f"Best rc: {rc[best_index]:.6g}")
    print(f"Best rt: {rt[best_index]:.6g}")
    print(f"Best SMEM fraction: {r_smem[best_index]:.6g}")
    print(f"SMEM: {smem_bytes[best_index] / 2**20:.3f} MiB")
    print(f"CUDA Cores: {int(cuda_cores[best_index])}")
    print(f"Tensor Cores: {int(tensor_cores[best_index])}")
    print(f"Total execution time: {total_time[best_index] * 1e3:.6f} ms")
    print(f"Effective throughput: {effective_flops / 1e12:.3f} TFLOP/s")
    for name, time_seconds in task_times.items():
        print(f"{name} time: {time_seconds[best_index] * 1e3:.6f} ms")
        traffic = task_traffic[name][best_index]
        print(f"{name} HBM traffic: {traffic / 2**20:.3f} MiB")
        print(
            f"{name} OI: "
            f"{task_operations_by_name[name] / traffic:.6f} FLOP/byte"
        )
    print(f"rmsnorm_square_reduction time: {rmsnorm_time[best_index] * 1e3:.6f} ms")
    print(
        "rmsnorm_square_reduction HBM traffic: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    print(f"activation time: {activation_time[best_index] * 1e3:.6f} ms")
    print(f"activation HBM traffic: {ACTIVATION_TASK.traffic_bytes / 2**20:.3f} MiB")
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
