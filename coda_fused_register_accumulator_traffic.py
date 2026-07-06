from __future__ import annotations

from coda_fused_traffic import (
    FusedTraffic,
    _traffic_for_pairwise_output_tile,
    _traffic_for_standard_output_tile,
    divisors,
)


# Outer-to-inner loop-order notation.  Keeping K innermost completes each
# output tile's K reduction while its accumulator is still live.
REGISTER_ACCUMULATOR_LOOP_ORDERS: tuple[tuple[str, str, str], ...] = (
    ("M", "N", "K"),
    ("N", "M", "K"),
)


def min_hbm_traffic_gemm_rms_scale(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    return _min_at_capacity(
        traffic_points_gemm_rms_scale(m, n, k, bytes_per_element),
        smem_bytes,
    )


def traffic_points_gemm_rms_scale(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, bytes_per_element)
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for n0 in divisors(n):
                for loop_order in REGISTER_ACCUMULATOR_LOOP_ORDERS:
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
                            final_tile_bytes=m0 * n0 * bytes_per_element,
                            aux_hbm_bytes=(m // m0)
                            * (n // n0)
                            * m0
                            * bytes_per_element,
                            aux_buffer_bytes=m0 * bytes_per_element,
                        )
                    )
    return points


def min_hbm_traffic_gemm_rms_swiglu(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    return _min_at_capacity(
        traffic_points_gemm_rms_swiglu(m, n, k, bytes_per_element),
        smem_bytes,
    )


def traffic_points_gemm_rms_swiglu(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, bytes_per_element)
    if n % 2 != 0:
        raise ValueError("n must be even for interleaved gate/up SwiGLU")

    p = n // 2
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for p0 in divisors(p):
                for loop_order in REGISTER_ACCUMULATOR_LOOP_ORDERS:
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
                            final_tile_bytes=m0 * p0 * bytes_per_element,
                            aux_hbm_bytes=(m // m0)
                            * (p // p0)
                            * m0
                            * bytes_per_element,
                            aux_buffer_bytes=m0 * bytes_per_element,
                        )
                    )
    return points


def min_hbm_traffic_gemm_residual_partial_rms_weight(
    m: int, n: int, k: int, smem_bytes: int, bytes_per_element: int
) -> int:
    return _min_at_capacity(
        traffic_points_gemm_residual_partial_rms_weight(
            m,
            n,
            k,
            bytes_per_element,
        ),
        smem_bytes,
    )


def traffic_points_gemm_residual_partial_rms_weight(
    m: int, n: int, k: int, bytes_per_element: int
) -> list[FusedTraffic]:
    _validate_problem(m, n, k, bytes_per_element)
    fp32_bytes = 4
    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        for k0 in divisors(k):
            for n0 in divisors(n):
                for loop_order in REGISTER_ACCUMULATOR_LOOP_ORDERS:
                    mt = m // m0
                    nt = n // n0
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
                            final_tile_bytes=m0 * n0 * bytes_per_element,
                            aux_hbm_bytes=(
                                mt * nt * m0 * n0 * bytes_per_element
                                + mt * nt * n0 * bytes_per_element
                                + mt * nt * m0 * fp32_bytes
                            ),
                            aux_buffer_bytes=(
                                m0 * n0 * bytes_per_element
                                + n0 * bytes_per_element
                                + m0 * fp32_bytes
                            ),
                        )
                    )
    return points


def min_hbm_traffic_down_weighted_sum_residual(
    m: int,
    n: int,
    k: int,
    smem_bytes: int,
    bytes_per_element: int,
    top_k: int = 1,
) -> int:
    return _min_at_capacity(
        traffic_points_down_weighted_sum_residual(
            m,
            n,
            k,
            bytes_per_element,
            top_k,
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
    _validate_problem(m, n, k, bytes_per_element)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if m % top_k != 0:
        raise ValueError("m must be divisible by top_k")

    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        if m0 % top_k != 0:
            continue
        token_m0 = m0 // top_k
        for k0 in divisors(k):
            for n0 in divisors(n):
                for loop_order in REGISTER_ACCUMULATOR_LOOP_ORDERS:
                    mt = m // m0
                    nt = n // n0
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
                            final_tile_bytes=token_m0 * n0 * bytes_per_element,
                            aux_hbm_bytes=(
                                mt * nt * m0 * bytes_per_element
                                + mt * nt * token_m0 * n0 * bytes_per_element
                            ),
                            aux_buffer_bytes=(
                                m0 * bytes_per_element
                                + token_m0 * n0 * bytes_per_element
                            ),
                        )
                    )
    return points


def _min_at_capacity(points: list[FusedTraffic], smem_bytes: int) -> int:
    if smem_bytes <= 0:
        raise ValueError("smem_bytes must be positive")
    best = min(
        (point.hbm_bytes for point in points if point.buffer_bytes <= smem_bytes),
        default=None,
    )
    if best is None:
        raise ValueError("no register-accumulator fused mapping fits in SMEM")
    return best


def _validate_problem(
    m: int,
    n: int,
    k: int,
    bytes_per_element: int,
) -> None:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
