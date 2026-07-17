"""Custom fused-kernel model — Fusion 5: SwiGLU activation + down.

Baseline-agnostic (``bl`` = decode or prefill module).

Fused kernel ``swiglu_down``: the down GEMM reads the gate+up tensor directly, applies
SwiGLU on chip to form the activated A tiles, then contracts over K=INTERMEDIATE.  The
activated tensor is never materialized.  This is a prologue fusion that widens down's A
input to 2*INTERMEDIATE (gate+up), so traffic is modeled with
``widened_a_points(a_width_mult=2)``.
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, widened_a_points

LABEL = "swiglu_down"

REMOVED_GEMM_STAGES = ("down",)
REMOVED_VECTOR_STAGES = ("activation",)


def kernel_dims(bl) -> tuple[int, int, int]:
    m = bl.padded_m(bl.TOKENS_PER_EXPERT, bl.TENSOR_CORE_MIN_BM)  # per-expert tokens
    n = bl.HIDDEN_SIZE                                            # 6144
    k = bl.INTERMEDIATE_SIZE                                      # 2048 (contraction)
    return m, n, k


def _count(bl) -> int:
    return bl.EXPERTS


def tensor_operations(bl) -> float:
    m, n, k = kernel_dims(bl)
    return 2 * m * n * k          # down GEMM


def cuda_operations(bl) -> float:
    # SwiGLU over the activated (INTERMEDIATE=K) elements, per expert.
    m, _, k = kernel_dims(bl)
    return m * k * bl.SWIGLU_FLOPS_PER_ELEMENT


def build_frontier(bl):
    m, n, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = widened_a_points(
        m,
        n,
        k,
        a_width_mult=2,                                # read gate+up (2*INTERMEDIATE)
        final_tile_bytes=lambda m0, n0: m0 * n0 * b,   # raw down output write
        aux_hbm_bytes=lambda m0, n0, mt, nt: 0,        # SwiGLU inline, no extra HBM
        aux_buffer_bytes=lambda m0, n0: 0,
    )
    return build_fused_frontier(
        label=LABEL,
        count=_count(bl),
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
