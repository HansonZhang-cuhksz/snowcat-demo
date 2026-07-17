"""Custom fused-kernel model — Fusion 6: up_gate + SwiGLU + down (full FFN fusion).

Baseline-agnostic (``bl`` = decode or prefill module).

The whole expert FFN as one GEMM-GEMM kernel: out = down(SwiGLU(up_gate(x))).  The gate+up
and activated intermediates never touch HBM.  Because down contracts over the full
INTERMEDIATE dim, the full ``activated[m0, :]`` row must be resident, so the kernel
processes an ``m0``-token row-block and reads **both** weight matrices once per block
(``mt = M/m0`` blocks -> weights read ``mt`` times).  Large ``m0`` avoids weight re-reads
but needs a big on-chip resident; small ``m0`` re-reads weights.  Enumerate ``m0`` to
capture this SMEM-gated tradeoff (decode M=64 fits; prefill M=32768/expert does not, so
the fusion re-reads weights heavily).  See ``plan.md`` for assumptions.
"""

from __future__ import annotations

from coda_fused_traffic import FusedTraffic, divisors

from fusion.common import build_fused_frontier

LABEL = "ffn_up_gate_swiglu_down"

REMOVED_GEMM_STAGES = ("up_gate", "down")
REMOVED_VECTOR_STAGES = ("activation",)


def kernel_dims(bl) -> tuple[int, int, int]:
    m = bl.padded_m(bl.TOKENS_PER_EXPERT, bl.TENSOR_CORE_MIN_BM)  # per-expert tokens
    hidden = bl.HIDDEN_SIZE
    intermediate = bl.INTERMEDIATE_SIZE
    return m, hidden, intermediate


def _count(bl) -> int:
    return bl.EXPERTS


def tensor_operations(bl) -> float:
    m, hidden, intermediate = kernel_dims(bl)
    up_gate = 2 * m * (2 * intermediate) * hidden
    down = 2 * m * hidden * intermediate
    return up_gate + down


def cuda_operations(bl) -> float:
    m, _, intermediate = kernel_dims(bl)
    return m * intermediate * bl.SWIGLU_FLOPS_PER_ELEMENT   # SwiGLU


def _points(bl) -> list[FusedTraffic]:
    m, hidden, intermediate = kernel_dims(bl)
    b = bl.BYTE_PER_ELEMENT
    w_ug = hidden * (2 * intermediate) * b       # up_gate weights
    w_dn = intermediate * hidden * b             # down weights
    x_out = 2 * m * hidden * b                   # x read + out write (once each)

    points: list[FusedTraffic] = []
    for m0 in divisors(m):
        if m0 < bl.TENSOR_CORE_MIN_BM:
            continue
        mt = m // m0
        traffic = x_out + mt * (w_ug + w_dn)
        buffer = (
            m0 * (intermediate + hidden) * b     # resident activated + out accumulators
            + (2 * intermediate) * b             # W_ug K-slice tile
            + hidden * b                         # W_dn K-slice tile
            + hidden * b                         # x row tile
        )
        points.append(
            FusedTraffic(
                buffer_bytes=buffer,
                hbm_bytes=traffic,
                bm=m0,
                bn=hidden,
                bk=intermediate,
                loop_order=("M", "N", "K"),
            )
        )
    return points


def build_frontier(bl):
    return build_fused_frontier(
        label=LABEL,
        count=_count(bl),
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=_points(bl),
        tile_filter=bl.tensor_core_tile_allowed,
    )
