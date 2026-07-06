from __future__ import annotations

from snowcat_demo.model.mapping import MappingPoint, enumerate_mappings
from snowcat_demo.model.pareto import best_at_capacity
from snowcat_demo.model.workload import GemmWorkload


DEFAULT_GEMM_MNK = (1024, 1024, 1024)
DEFAULT_BYTES_PER_ELEMENT = 2

# When M, N, and K all have more than one tile, these are the only loop orders
# that complete each output tile's K reduction before moving to another output
# tile.  The order notation is outer-to-inner, so K is innermost.
FULLY_TILED_REGISTER_ACCUMULATOR_LOOP_ORDERS: tuple[tuple[str, str, str], ...] = (
    ("M", "N", "K"),
    ("N", "M", "K"),
)


def output_accumulator_stays_in_registers(
    workload: GemmWorkload,
    m0: int,
    k0: int,
    n0: int,
    loop_order: tuple[str, str, str],
) -> bool:
    """Return True if a mapping completes each output tile before eviction.

    Snowcat loop orders are written outer-to-inner.  Register-resident output
    accumulation requires the K loop to be inside all varying output-tile loops
    (M and N).  If a dimension has only one tile, its loop position does not
    matter because it does not force a switch to a different output tile.
    """
    mt, kt, nt = workload.tile_counts(m0, k0, n0)
    extents = {"M": mt, "K": kt, "N": nt}
    varying_output_positions = [
        loop_order.index(dim) for dim in ("M", "N") if extents[dim] > 1
    ]
    deepest_output_position = max(varying_output_positions, default=-1)
    return loop_order.index("K") > deepest_output_position


def enumerate_register_accumulator_mappings(
    workload: GemmWorkload,
) -> list[MappingPoint]:
    """Enumerate Snowcat mappings that avoid output accumulator spill/reload."""
    points = []
    for point in enumerate_mappings(workload):
        mapping = point.mapping
        if output_accumulator_stays_in_registers(
            workload,
            mapping.m0,
            mapping.k0,
            mapping.n0,
            mapping.loop_order,
        ):
            points.append(point)
    return points


def min_attainable_traffic_register_accumulator(
    gemm_mnk: tuple[int, int, int],
    sram_capacity_bytes: int,
    bytes_per_element: int = DEFAULT_BYTES_PER_ELEMENT,
) -> int:
    """Return min HBM traffic for GEMM mappings with register C accumulation.

    This mirrors ``ski_slope.min_attainable_traffic`` but filters the mapspace
    to schedules that keep the output accumulator tile live until all K tiles
    have been consumed.  The SRAM capacity still means one A tile, one W tile,
    and one output accumulator tile.
    """
    m, n, k = gemm_mnk
    workload = GemmWorkload(
        m=m,
        k=k,
        n=n,
        bytes_per_element=bytes_per_element,
    )
    best = best_at_capacity(
        enumerate_register_accumulator_mappings(workload),
        sram_capacity_bytes,
    )
    if best is None:
        raise ValueError("no register-accumulator mapping fits in SMEM")
    return best.backing_store_bytes


def ski_slope_points_register_accumulator(
    gemm_mnk: tuple[int, int, int],
    bytes_per_element: int = DEFAULT_BYTES_PER_ELEMENT,
) -> list[tuple[int, int]]:
    m, n, k = gemm_mnk
    workload = GemmWorkload(
        m=m,
        k=k,
        n=n,
        bytes_per_element=bytes_per_element,
    )
    mappings = enumerate_register_accumulator_mappings(workload)
    capacities = sorted({point.buffer_bytes for point in mappings})

    points: list[tuple[int, int]] = []
    for capacity in capacities:
        best = best_at_capacity(mappings, capacity)
        if best is not None:
            points.append((capacity, best.backing_store_bytes))
    return points
