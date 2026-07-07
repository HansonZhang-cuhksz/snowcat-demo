from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expert_distribution import ExpertTokenDistribution, binomial_expert_token_distribution
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
BATCH_TOKENS = 4096
EXPERTS = 256
TOKENS_PER_EXPERT = 128
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
ROUTER_TOP_K = EXPERTS * TOKENS_PER_EXPERT // BATCH_TOKENS
TENSOR_CORE_MIN_BM = 16
TENSOR_CORE_MIN_BN = 8
TENSOR_CORE_MIN_BK = 16

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

# False preserves the original even-routing estimate: every expert receives
# TOKENS_PER_EXPERT tokens.  True uses an expected-value random-routing model
# where each expert's token count follows Binomial(BATCH_TOKENS, top_k/EXPERTS).
USE_RANDOM_EXPERT_DISTRIBUTION = False
EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF = 1e-12


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
    bm: np.ndarray
    bn: np.ndarray
    bk: np.ndarray
    loop_orders: tuple[tuple[str, str, str], ...]


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


def expert_token_distribution() -> ExpertTokenDistribution:
    return binomial_expert_token_distribution(
        batch_tokens=BATCH_TOKENS,
        experts=EXPERTS,
        top_k=ROUTER_TOP_K,
        probability_cutoff=EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF,
    )


def group_random_expert_gemm_tasks(
    distribution: ExpertTokenDistribution,
) -> tuple[list[GemmTaskGroup], dict[str, float], dict[str, str]]:
    task_groups = [
        GemmTaskGroup(
            label="router",
            task=GemmTask("router", BATCH_TOKENS, EXPERTS, HIDDEN_SIZE),
            count=1,
        )
    ]
    expert_weights = {"router": 1.0}
    aggregate_names = {"router": "router"}

    for task_name, n, k in (
        ("up_gate", 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ("down", HIDDEN_SIZE, INTERMEDIATE_SIZE),
    ):
        aggregate_name = f"{task_name}_random_expected"
        for tokens, probability in distribution.support:
            if tokens == 0:
                continue
            label = f"{task_name}_m{tokens}"
            task_groups.append(
                GemmTaskGroup(
                    label=label,
                    task=GemmTask(task_name, tokens, n, k),
                    count=1,
                )
            )
            expert_weights[label] = EXPERTS * probability
            aggregate_names[label] = aggregate_name

    return task_groups, expert_weights, aggregate_names


def tensor_core_tile_allowed(bm: int, bn: int, bk: int) -> bool:
    return (
        bm >= TENSOR_CORE_MIN_BM
        and bn >= TENSOR_CORE_MIN_BN
        and bk >= TENSOR_CORE_MIN_BK
    )


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
    mapping_points = [
        point
        for point in mapping_points
        if tensor_core_tile_allowed(
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
        )
    ]
    if not mapping_points:
        raise ValueError(
            f"no tensor-core-compatible mapping for {group.label}; "
            f"requires BM>={TENSOR_CORE_MIN_BM}, "
            f"BN>={TENSOR_CORE_MIN_BN}, BK>={TENSOR_CORE_MIN_BK}"
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
        for point in mapping_points
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
        next_buffer_index = len(frontier_buffer_list)
        if next_buffer_index > 0 and buffer_bytes == frontier_buffer_list[-1]:
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

    return TrafficFrontier(
        label=group.label,
        count=group.count,
        operations=group.task.operations,
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


def output_paths() -> tuple[str, str, str]:
    suffix_parts = []
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        suffix_parts.append("register_accumulator")
    if USE_RANDOM_EXPERT_DISTRIBUTION:
        suffix_parts.append("random_experts")
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    return (
        f"./result/ffn_area{suffix}_times.csv",
        f"./result/ffn_area{suffix}_total_time.png",
        f"./result/ffn_area{suffix}_activation_time.png",
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


def selected_mapping_from_frontier(
    frontier: TrafficFrontier, capacity_bytes: float
) -> tuple[int, int, int, tuple[str, str, str], int] | None:
    index = np.searchsorted(frontier.buffer_bytes, capacity_bytes, side="right") - 1
    if index < 0:
        return None
    return (
        int(frontier.bm[index]),
        int(frontier.bn[index]),
        int(frontier.bk[index]),
        frontier.loop_orders[index],
        int(frontier.buffer_bytes[index]),
    )


def format_selected_mapping(frontier: TrafficFrontier, capacity_bytes: float) -> str:
    mapping = selected_mapping_from_frontier(frontier, capacity_bytes)
    if mapping is None:
        return "no mapping fits selected SMEM capacity"
    bm, bn, bk, loop_order, stage_smem_bytes = mapping
    return (
        f"BM={bm}, BN={bn}, BK={bk}, "
        f"loop_order={'-'.join(loop_order)}, "
        f"stage_smem={stage_smem_bytes / 2**20:.6f} MiB "
        f"({stage_smem_bytes} bytes)"
    )


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


def total_hbm_traffic_bytes(task_traffic: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros_like(next(iter(task_traffic.values())), dtype=float)
    for traffic_bytes in task_traffic.values():
        total = total + traffic_bytes
    if INCLUDE_RMSNORM:
        total = total + RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes
    total = total + ACTIVATION_TASK.traffic_bytes
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
            "total_hbm_mib",
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
        total_hbm_traffic = total_hbm_traffic_bytes(task_traffic)

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


task_operations_by_name: dict[str, float] = {}


def main() -> None:
    if ROUTER_TOP_K * BATCH_TOKENS != EXPERTS * TOKENS_PER_EXPERT:
        raise ValueError("Expert token count must equal batch tokens times top-k")

    task_operations_by_name.clear()

    rc, rt, r_smem = make_area_grid(AREA_GRID_STEP)
    smem_bytes = r_smem * A_total / A_bit / 8
    cuda_cores = np.floor(rc * A_total / A_cuda_core)
    tensor_cores = np.floor(rt * A_total / A_tensor_core)
    cuda_roof = cuda_cores * ACTIVATION_FLOPS_PER_CUDA_CORE
    tensor_roof = tensor_cores * TENSOR_FLOPS
    flops_per_cuda_core_cycle = ACTIVATION_FLOPS_PER_CUDA_CORE / CUDA_CLOCK_HZ

    distribution = (
        expert_token_distribution() if USE_RANDOM_EXPERT_DISTRIBUTION else None
    )
    if distribution is None:
        task_groups = group_gemm_tasks(GEMM_TASKS)
        group_weights = {group.label: 1.0 for group in task_groups}
        aggregate_names = {group.label: group.label for group in task_groups}
    else:
        task_groups, group_weights, aggregate_names = group_random_expert_gemm_tasks(
            distribution
        )

    frontiers, frontier_mode = build_frontiers(task_groups)
    gemm_operations = 0.0
    task_times: dict[str, np.ndarray] = {}
    task_traffic: dict[str, np.ndarray] = {}
    for frontier in frontiers:
        aggregate_name = aggregate_names[frontier.label]
        weight = group_weights[frontier.label]
        weighted_operations = weight * frontier.count * frontier.operations
        gemm_operations += weighted_operations
        task_operations_by_name[aggregate_name] = (
            task_operations_by_name.get(aggregate_name, 0.0) + weighted_operations
        )

        weighted_time = weight * gemm_time_from_frontier(
            frontier, smem_bytes, tensor_roof
        )
        weighted_traffic = (
            weight * frontier.count * min_traffic_from_frontier(frontier, smem_bytes)
        )
        if aggregate_name in task_times:
            task_times[aggregate_name] = task_times[aggregate_name] + weighted_time
            task_traffic[aggregate_name] = (
                task_traffic[aggregate_name] + weighted_traffic
            )
        else:
            task_times[aggregate_name] = weighted_time
            task_traffic[aggregate_name] = weighted_traffic

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
    total_hbm_traffic = total_hbm_traffic_bytes(task_traffic)

    print("\n=== Configuration ===")
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
    print(f"Activation OI: {ACTIVATION_TASK.operational_intensity:.6f} FLOP/byte")
    print(
        "Expert weighted-sum OI: "
        f"{EXPERT_WEIGHTED_SUM_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Residual add OI: {RESIDUAL_ADD_TASK.operational_intensity:.6f} FLOP/byte")

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
    frontiers_by_aggregate: dict[str, list[TrafficFrontier]] = {}
    for frontier in frontiers:
        aggregate_name = aggregate_names[frontier.label]
        frontiers_by_aggregate.setdefault(aggregate_name, []).append(frontier)

    print("\n=== GEMM Stages ===")
    for name, time_seconds in task_times.items():
        print(f"\n{name}")
        print(f"  time: {time_seconds[best_index] * 1e3:.6f} ms")
        traffic = task_traffic[name][best_index]
        print(f"  HBM traffic: {traffic / 2**20:.3f} MiB")
        print(
            "  OI: "
            f"{task_operations_by_name[name] / traffic:.6f} FLOP/byte"
        )
        stage_frontiers = frontiers_by_aggregate.get(name, [])
        if len(stage_frontiers) == 1:
            print(
                "  mapping: "
                f"{format_selected_mapping(stage_frontiers[0], smem_bytes[best_index])}"
            )
        elif stage_frontiers:
            print("  constituent mappings:")
            for stage_frontier in stage_frontiers:
                weight = group_weights[stage_frontier.label]
                print(
                    f"    {stage_frontier.label} "
                    f"(expected_count={weight:.8g}): "
                    f"{format_selected_mapping(stage_frontier, smem_bytes[best_index])}"
                )

    print("\n=== Vector / Reduction Stages ===")
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
