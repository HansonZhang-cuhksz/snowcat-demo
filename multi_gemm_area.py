from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


# Chip constants
A_total = 694.116 * 10**6              # um^2
A_bit = 0.0864                         # um^2/bit
logic_density = 39.98                  # MTr/mm^2
tensor_logic = 6                       # MTr/tensor
A_tensor = tensor_logic / logic_density * 10**6  # um^2/tensor

tensor_flops = 512 * 1.00 * 10**9      # flops/s/tensor
bw = 2.04 * 10**12                     # byte/s
BYTE_PER_ELEMENT = 2
CPU_WORKERS = 32
PARALLEL_FRONTIER_MIN_GROUPS = CPU_WORKERS


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
class TrafficFrontier:
    label: str
    count: int
    operations: int
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray


# Edit this list to model a layer, block, or full sequence of GEMMs.
GEMM_TASKS = [GemmTask("router", 8192, 256, 6144)] + [GemmTask("up_gate", 128, 4096, 6144)] * 512 + [GemmTask("down", 128, 6144, 2048)] * 512


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

    return TrafficFrontier(
        label=group.label,
        count=group.count,
        operations=group.task.operations,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
    )


def min_traffic_from_frontier(
    frontier: TrafficFrontier, capacities: np.ndarray
) -> np.ndarray:
    indexes = np.searchsorted(frontier.buffer_bytes, capacities, side="right") - 1
    min_traffic = np.full(len(capacities), np.nan, dtype=float)
    valid = indexes >= 0
    min_traffic[valid] = frontier.traffic_bytes[indexes[valid]]
    return min_traffic


def execution_time_from_frontier(
    frontier: TrafficFrontier, capacities: np.ndarray, compute_roof: np.ndarray
) -> np.ndarray:
    min_traffic = min_traffic_from_frontier(frontier, capacities)
    oi = frontier.operations / min_traffic
    memory_roof = oi * bw
    peak = np.minimum(memory_roof, compute_roof)

    time_seconds = np.full(len(capacities), np.nan, dtype=float)
    np.divide(frontier.operations, peak, out=time_seconds, where=peak > 0)
    return frontier.count * time_seconds


def build_frontiers(task_groups: list[GemmTaskGroup]) -> tuple[list[TrafficFrontier], str]:
    if len(task_groups) < PARALLEL_FRONTIER_MIN_GROUPS:
        return [build_traffic_frontier(group) for group in task_groups], "serial"

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
        frontiers = list(
            executor.map(build_traffic_frontier, task_groups, chunksize=1)
        )
    return frontiers, f"{CPU_WORKERS}-process"


def write_csv(
    path: str,
    r: np.ndarray,
    sram_bytes: np.ndarray,
    tensors: np.ndarray,
    task_groups: list[GemmTaskGroup],
    task_times: dict[str, np.ndarray],
    total_time: np.ndarray,
) -> None:
    total_operations = sum(group.operations for group in task_groups)
    effective_flops = total_operations / total_time

    with open(path, "w", newline="") as csvfile:
        fieldnames = [
            "r",
            "sram_mib",
            "tensor_cores",
            "total_time_ms",
            "effective_tflops",
            *[f"{name}_time_ms" for name in task_times],
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(len(r)):
            row = {
                "r": r[index],
                "sram_mib": sram_bytes[index] / 2**20,
                "tensor_cores": tensors[index],
                "total_time_ms": total_time[index] * 1e3,
                "effective_tflops": effective_flops[index] / 1e12,
            }
            for name, time_seconds in task_times.items():
                row[f"{name}_time_ms"] = time_seconds[index] * 1e3
            writer.writerow(row)


def plot_results(
    r: np.ndarray,
    task_groups: list[GemmTaskGroup],
    task_times: dict[str, np.ndarray],
    total_time: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("result").mkdir(exist_ok=True)
    total_operations = sum(group.operations for group in task_groups)
    effective_flops = total_operations / total_time

    plt.figure(figsize=(10, 6))
    plt.plot(r, total_time * 1e3, label="Total execution time")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Execution Time (ms)")
    plt.title("Total GEMM Execution Time vs. SRAM Utilization")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("./result/multi_gemm_total_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, time_seconds in task_times.items():
        plt.plot(r, time_seconds * 1e3, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Execution Time (ms)")
    plt.title("Individual GEMM Execution Time vs. SRAM Utilization")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("./result/multi_gemm_individual_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(r, effective_flops / 1e12, label="Effective throughput")
    plt.xscale("log")
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Effective Throughput (TFLOP/s)")
    plt.title("Total GEMM Effective Throughput vs. SRAM Utilization")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("./result/multi_gemm_effective_tflops.png", dpi=160)
    plt.close()


def main() -> None:
    r = np.unique(
        np.concatenate(
            (
                np.geomspace(1e-8, 1e-3, 300),
                np.linspace(1e-3, 0.999, 700),
            )
        )
    )

    sram_bytes = r * A_total / A_bit / 8
    tensors = np.floor((1 - r) * A_total / A_tensor)
    compute_roof = tensors * tensor_flops

    task_groups = group_gemm_tasks(GEMM_TASKS)
    frontiers, frontier_mode = build_frontiers(task_groups)
    task_times = {
        frontier.label: execution_time_from_frontier(
            frontier,
            sram_bytes,
            compute_roof,
        )
        for frontier in frontiers
    }
    total_time = np.sum(np.array(list(task_times.values())), axis=0)

    write_csv(
        "./result/multi_gemm_times.csv",
        r,
        sram_bytes,
        tensors,
        task_groups,
        task_times,
        total_time,
    )
    plot_results(r, task_groups, task_times, total_time)

    best_index = int(np.nanargmin(total_time))
    total_operations = sum(group.operations for group in task_groups)
    effective_flops = total_operations / total_time[best_index]
    print(f"CPU workers available: {CPU_WORKERS}")
    print(f"Frontier build mode: {frontier_mode}")
    print(f"Unique GEMM groups: {len(task_groups)}")
    print(f"Total traffic frontier points: {sum(len(frontier.buffer_bytes) for frontier in frontiers)}")
    print(f"Best r: {r[best_index]:.6g}")
    print(f"SRAM: {sram_bytes[best_index] / 2**20:.3f} MiB")
    print(f"Tensor Cores: {int(tensors[best_index])}")
    print(f"Total execution time: {total_time[best_index] * 1e3:.6f} ms")
    print(f"Effective throughput: {effective_flops / 1e12:.3f} TFLOP/s")


if __name__ == "__main__":
    main()
