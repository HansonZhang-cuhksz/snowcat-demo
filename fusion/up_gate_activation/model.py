"""Custom fused-kernel model — Fusion 4: up_gate + SwiGLU activation.

Baseline-agnostic (``bl`` = decode or prefill module).

Fused kernel ``up_gate_swiglu``: the up_gate GEMM produces interleaved gate/up tiles on
chip and applies SwiGLU in the epilogue, writing only the activated [M, INTERMEDIATE]
output (CODA pairwise output tile).  The raw gate+up is never written / re-read.
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, pairwise_tile_points

LABEL = "up_gate_swiglu"

REMOVED_GEMM_STAGES = ("up_gate",)
REMOVED_VECTOR_STAGES = ("activation",)


def kernel_dims(bl) -> tuple[int, int, int]:
    """(M, p, K): the GEMM produces 2*p = 2*INTERMEDIATE interleaved gate/up cols."""
    m = bl.padded_m(bl.TOKENS_PER_EXPERT, bl.TENSOR_CORE_MIN_BM)  # per-expert tokens
    p = bl.INTERMEDIATE_SIZE                                      # 2048 (activated cols)
    k = bl.HIDDEN_SIZE                                            # 6144
    return m, p, k


def _count(bl) -> int:
    return bl.EXPERTS


def tensor_operations(bl) -> float:
    m, p, k = kernel_dims(bl)
    return 2 * m * (2 * p) * k          # up_gate GEMM (gate+up)


def cuda_operations(bl) -> float:
    m, p, _ = kernel_dims(bl)
    return m * p * bl.SWIGLU_FLOPS_PER_ELEMENT   # SwiGLU over activated elements


def build_frontier(bl):
    m, p, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = pairwise_tile_points(
        m,
        p,
        k,
        final_tile_bytes=lambda m0, p0: m0 * p0 * b,   # write activated output only
        aux_hbm_bytes=lambda m0, p0, mt, pt: 0,        # SwiGLU elementwise on-chip
        aux_buffer_bytes=lambda m0, p0: 0,
    )
    return build_fused_frontier(
        label=LABEL,
        count=_count(bl),
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
