from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from snowcat_demo.model.mapping import MappingPoint, enumerate_mappings
from snowcat_demo.model.pareto import best_at_capacity
from snowcat_demo.model.workload import GemmWorkload


DEFAULT_GEMM_MNK = (128, 4096, 4096)
DEFAULT_BYTES_PER_ELEMENT = 2


def _import_pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "snowcat-matplotlib")
    )
    import matplotlib.pyplot as plt

    return plt


def _workload_from_mnk(
    gemm_mnk: tuple[int, int, int], bytes_per_element: int
) -> GemmWorkload:
    m, n, k = gemm_mnk
    return GemmWorkload(m=m, k=k, n=n, bytes_per_element=bytes_per_element)


def min_attainable_traffic(
    gemm_mnk: tuple[int, int, int],
    sram_capacity_bytes: int,
    bytes_per_element: int = DEFAULT_BYTES_PER_ELEMENT,
) -> int:
    """Return the minimum backing-store traffic for GEMM (M, N, K).

    The search uses Snowcat's current mapspace: divisor tile sizes for M/K/N
    and all loop orders. SRAM capacity is interpreted as the capacity needed
    to hold one A tile, one W tile, and one B tile.
    """
    workload = _workload_from_mnk(gemm_mnk, bytes_per_element)
    best = best_at_capacity(enumerate_mappings(workload), sram_capacity_bytes)
    if best is None:
        raise ValueError("no mapping fits in the requested SRAM capacity")
    return best.backing_store_bytes


def ski_slope_points(
    gemm_mnk: tuple[int, int, int],
    bytes_per_element: int = DEFAULT_BYTES_PER_ELEMENT,
) -> list[tuple[int, int]]:
    workload = _workload_from_mnk(gemm_mnk, bytes_per_element)
    mappings = enumerate_mappings(workload)
    capacities = sorted({point.buffer_bytes for point in mappings})

    points: list[tuple[int, int]] = []
    for capacity in capacities:
        best = best_at_capacity(mappings, capacity)
        if best is not None:
            points.append((capacity, best.backing_store_bytes))
    return points


def plot_ski_slope(
    gemm_mnk: tuple[int, int, int],
    bytes_per_element: int = DEFAULT_BYTES_PER_ELEMENT,
    output_path: Path | None = None,
) -> list[tuple[int, int]]:
    workload = _workload_from_mnk(gemm_mnk, bytes_per_element)
    mappings = enumerate_mappings(workload)
    capacities = sorted({point.buffer_bytes for point in mappings})

    slope: list[MappingPoint] = []
    for capacity in capacities:
        best = best_at_capacity(mappings, capacity)
        if best is not None:
            slope.append(best)

    x = [point.buffer_bytes for point in slope]
    y = [point.backing_store_bytes for point in slope]

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(x, y, where="post", linewidth=2, label="best attainable traffic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SRAM capacity (bytes)")
    ax.set_ylabel("Minimum attainable backing-store traffic (bytes)")
    ax.set_title(f"Snowcat ski slope: GEMM M={gemm_mnk[0]}, N={gemm_mnk[1]}, K={gemm_mnk[2]}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    return list(zip(x, y, strict=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Snowcat GEMM ski slope.")
    parser.add_argument("--m", type=int, default=DEFAULT_GEMM_MNK[0])
    parser.add_argument("--n", type=int, default=DEFAULT_GEMM_MNK[1])
    parser.add_argument("--k", type=int, default=DEFAULT_GEMM_MNK[2])
    parser.add_argument("--bytes-per-element", type=int, default=DEFAULT_BYTES_PER_ELEMENT)
    parser.add_argument("--output", type=Path, default=Path("ski_slope.png"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gemm_mnk = (args.m, args.n, args.k)
    points = plot_ski_slope(
        gemm_mnk=gemm_mnk,
        bytes_per_element=args.bytes_per_element,
        output_path=args.output,
    )
    print(f"plotted {len(points):,} SRAM capacity points")
    print(f"plot written to: {args.output}")


if __name__ == "__main__":
    main()
