from __future__ import annotations

from dataclasses import dataclass
from itertools import product


LOOP_ORDERS: tuple[tuple[str, str, str], ...] = (
    ("M", "K", "N"),
    ("M", "N", "K"),
    ("K", "M", "N"),
    ("K", "N", "M"),
    ("N", "M", "K"),
    ("N", "K", "M"),
)


@dataclass(frozen=True)
class FusedTraffic:
    buffer_bytes: int
    hbm_bytes: int


def divisors(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")

    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= n:
        if n % candidate == 0:
            small.append(candidate)
            paired = n // candidate
            if paired != candidate:
                large.append(paired)
        candidate += 1
    return small + large[::-1]


def min_hbm_traffic_gemm_rms_scale(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    """Minimum HBM traffic for GEMM with a row-wise RMS scale epilogue.

    Computes O = (A @ B) * r[:, None].  This models CODA's delayed RMSNorm
    scale for router/QKV-style projections.  The row scale is loaded once per
    output tile because the epilogue sees one MxN tile at a time.
    """
    return _min_at_capacity(
        traffic_points_gemm_rms_scale(m, n, k, bytes_per_element), smem_bytes
    )


def traffic_points_gemm_rms_scale(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    return _points_over_standard_tiles(
        m,
        n,
        k,
        bytes_per_element,
        final_tile_bytes=lambda m0, n0: m0 * n0 * bytes_per_element,
        aux_hbm_bytes=lambda m0, n0, mt, nt: mt * nt * m0 * bytes_per_element,
        aux_buffer_bytes=lambda m0, n0: m0 * bytes_per_element,
    )


def min_hbm_traffic_gemm_rms_swiglu(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    """Minimum HBM traffic for GEMM with row-wise RMS scale and SwiGLU.

    The GEMM produces N columns, arranged as interleaved gate/up pairs, and the
    epilogue stores N/2 activated columns.  Raw gate/up tiles that would be
    written and reloaded by an unfused activation are never materialized.
    """
    return _min_at_capacity(
        traffic_points_gemm_rms_swiglu(m, n, k, bytes_per_element), smem_bytes
    )


def traffic_points_gemm_rms_swiglu(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, 1, bytes_per_element)
    if n % 2 != 0:
        raise ValueError("n must be even for interleaved gate/up SwiGLU")

    p = n // 2
    points: list[FusedTraffic] = []
    for m0, k0, p0, loop_order in product(
        divisors(m), divisors(k), divisors(p), LOOP_ORDERS
    ):
        point = _traffic_for_pairwise_output_tile(
            m=m,
            p=p,
            k=k,
            m0=m0,
            p0=p0,
            k0=k0,
            loop_order=loop_order,
            bytes_per_element=bytes_per_element,
            final_tile_bytes=m0 * p0 * bytes_per_element,
            aux_hbm_bytes=m // m0 * (p // p0) * m0 * bytes_per_element,
            aux_buffer_bytes=m0 * bytes_per_element,
        )
        points.append(point)
    return points


def min_hbm_traffic_gemm_residual_partial_rms_weight(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    """Minimum HBM traffic for GEMM + residual + partial RMS + gamma scale.

    Computes D = A @ B + C, stores O = D * gamma, and writes one FP32 partial
    sum-of-squares per output row per N tile.  This is CODA's first half of the
    GEMM-Residual-RMSNorm-GEMM reparameterization.
    """
    return _min_at_capacity(
        traffic_points_gemm_residual_partial_rms_weight(
            m, n, k, bytes_per_element
        ),
        smem_bytes,
    )


def traffic_points_gemm_residual_partial_rms_weight(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    fp32_bytes = 4
    return _points_over_standard_tiles(
        m,
        n,
        k,
        bytes_per_element,
        final_tile_bytes=lambda m0, n0: m0 * n0 * bytes_per_element,
        aux_hbm_bytes=lambda m0, n0, mt, nt: (
            mt * nt * m0 * n0 * bytes_per_element  # residual C tile reads
            + mt * nt * n0 * bytes_per_element  # gamma vector reads
            + mt * nt * m0 * fp32_bytes  # partial RMS-stat writes
        ),
        aux_buffer_bytes=lambda m0, n0: (
            m0 * n0 * bytes_per_element
            + n0 * bytes_per_element
            + m0 * fp32_bytes
        ),
    )


def min_hbm_traffic_down_weighted_sum_residual(
    m: int,
    n: int,
    k: int,
    smem_bytes: int,
    bytes_per_element: int,
    top_k: int = 1,
) -> int:
    """Minimum HBM traffic for an ideal down-GEMM + combine + residual fusion.

    The GEMM has M expert-token rows.  For top_k > 1, every top_k GEMM rows are
    assumed to belong to one original token and are accumulated on chip before a
    single residual-added output row is stored.  This is an optimistic lower
    bound for a full MoE top-k combine fusion; if rows are not co-scheduled, the
    combine must spill to HBM and this estimate is too low.
    """
    return _min_at_capacity(
        traffic_points_down_weighted_sum_residual(
            m, n, k, bytes_per_element, top_k
        ),
        smem_bytes,
    )


def traffic_points_down_weighted_sum_residual(
    m: int,
    n: int,
    k: int,
    bytes_per_element: int,
    top_k: int = 1,
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, 1, bytes_per_element)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if m % top_k != 0:
        raise ValueError("m must be divisible by top_k")

    points: list[FusedTraffic] = []
    for m0, k0, n0, loop_order in product(
        divisors(m), divisors(k), divisors(n), LOOP_ORDERS
    ):
        if m0 % top_k != 0:
            continue
        token_m0 = m0 // top_k
        point = _traffic_for_standard_output_tile(
            m=m,
            n=n,
            k=k,
            m0=m0,
            n0=n0,
            k0=k0,
            loop_order=loop_order,
            bytes_per_element=bytes_per_element,
            final_tile_bytes=token_m0 * n0 * bytes_per_element,
            aux_hbm_bytes=(
                (m // m0) * (n // n0) * m0 * bytes_per_element
                + (m // m0) * (n // n0) * token_m0 * n0 * bytes_per_element
            ),
            aux_buffer_bytes=(
                m0 * bytes_per_element + token_m0 * n0 * bytes_per_element
            ),
        )
        points.append(point)
    return points

def _points_over_standard_tiles(
    m: int,
    n: int,
    k: int,
    bytes_per_element: int,
    final_tile_bytes,
    aux_hbm_bytes,
    aux_buffer_bytes,
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, 1, bytes_per_element)
    points: list[FusedTraffic] = []
    for m0, k0, n0, loop_order in product(
        divisors(m), divisors(k), divisors(n), LOOP_ORDERS
    ):
        point = _traffic_for_standard_output_tile(
            m=m,
            n=n,
            k=k,
            m0=m0,
            n0=n0,
            k0=k0,
            loop_order=loop_order,
            bytes_per_element=bytes_per_element,
            final_tile_bytes=final_tile_bytes(m0, n0),
            aux_hbm_bytes=aux_hbm_bytes(m0, n0, m // m0, n // n0),
            aux_buffer_bytes=aux_buffer_bytes(m0, n0),
        )
        points.append(point)
    return points


def _min_at_capacity(points: list[FusedTraffic], smem_bytes: int) -> int:
    if smem_bytes <= 0:
        raise ValueError("smem_bytes must be positive")
    best = min(
        (point.hbm_bytes for point in points if point.buffer_bytes <= smem_bytes),
        default=None,
    )
    if best is None:
        raise ValueError("no fused GEMM mapping fits in SMEM")
    return best


def _traffic_for_standard_output_tile(
    m: int,
    n: int,
    k: int,
    m0: int,
    n0: int,
    k0: int,
    loop_order: tuple[str, str, str],
    bytes_per_element: int,
    final_tile_bytes: int,
    aux_hbm_bytes: int,
    aux_buffer_bytes: int,
) -> FusedTraffic:
    mt = m // m0
    nt = n // n0
    kt = k // k0
    a_tile_bytes = m0 * k0 * bytes_per_element
    w_tile_bytes = k0 * n0 * bytes_per_element
    raw_output_tile_bytes = m0 * n0 * bytes_per_element
    output_tiles = mt * nt

    extents = {"M": mt, "K": kt, "N": nt}
    a_reads = _run_count(loop_order, extents, ("M", "K")) * a_tile_bytes
    w_reads = _run_count(loop_order, extents, ("K", "N")) * w_tile_bytes
    partial_reads, partial_writes = _partial_accumulator_traffic(
        loop_order, extents, output_tiles, raw_output_tile_bytes
    )

    return FusedTraffic(
        buffer_bytes=(
            a_tile_bytes + w_tile_bytes + raw_output_tile_bytes + aux_buffer_bytes
        ),
        hbm_bytes=(
            a_reads
            + w_reads
            + partial_reads
            + partial_writes
            + output_tiles * final_tile_bytes
            + aux_hbm_bytes
        ),
    )


def _traffic_for_pairwise_output_tile(
    m: int,
    p: int,
    k: int,
    m0: int,
    p0: int,
    k0: int,
    loop_order: tuple[str, str, str],
    bytes_per_element: int,
    final_tile_bytes: int,
    aux_hbm_bytes: int,
    aux_buffer_bytes: int,
) -> FusedTraffic:
    mt = m // m0
    pt = p // p0
    kt = k // k0
    a_tile_bytes = m0 * k0 * bytes_per_element
    w_tile_bytes = k0 * (2 * p0) * bytes_per_element
    raw_output_tile_bytes = m0 * (2 * p0) * bytes_per_element
    output_tiles = mt * pt

    extents = {"M": mt, "K": kt, "N": pt}
    a_reads = _run_count(loop_order, extents, ("M", "K")) * a_tile_bytes
    w_reads = _run_count(loop_order, extents, ("K", "N")) * w_tile_bytes
    partial_reads, partial_writes = _partial_accumulator_traffic(
        loop_order, extents, output_tiles, raw_output_tile_bytes
    )

    return FusedTraffic(
        buffer_bytes=(
            a_tile_bytes + w_tile_bytes + raw_output_tile_bytes + aux_buffer_bytes
        ),
        hbm_bytes=(
            a_reads
            + w_reads
            + partial_reads
            + partial_writes
            + output_tiles * final_tile_bytes
            + aux_hbm_bytes
        ),
    )


def _partial_accumulator_traffic(
    loop_order: tuple[str, str, str],
    extents: dict[str, int],
    output_tiles: int,
    raw_output_tile_bytes: int,
) -> tuple[int, int]:
    varying_output_positions = [
        loop_order.index(dim) for dim in ("M", "N") if extents[dim] > 1
    ]
    deepest_output_position = max(varying_output_positions, default=-1)
    k_is_inside_output_run = loop_order.index("K") > deepest_output_position
    kt = extents["K"]

    if k_is_inside_output_run:
        return 0, 0
    partial_bytes = output_tiles * max(kt - 1, 0) * raw_output_tile_bytes
    return partial_bytes, partial_bytes


def _run_count(
    loop_order: tuple[str, str, str],
    extents: dict[str, int],
    key_dims: tuple[str, ...],
) -> int:
    varying_key_positions = [
        loop_order.index(dim) for dim in key_dims if extents[dim] > 1
    ]
    if not varying_key_positions:
        return 1
    deepest_key_position = max(varying_key_positions)
    count = 1
    for dim in loop_order[: deepest_key_position + 1]:
        count *= extents[dim]
    return count


def _validate_problem(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> None:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if smem_bytes <= 0:
        raise ValueError("smem_bytes must be positive")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
