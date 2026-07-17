"""Custom fused-kernel model — Fusion 3: pre-FFN RMSNorm + up_gate.

Baseline-agnostic (``bl`` = decode or prefill module).

Fused kernel ``up_gate_rmsnorm``: up_gate computes the per-row RMS inline while streaming
its K (=HIDDEN) reduction of the A-input it already reads, applying the per-row scale at
the epilogue (gamma folded into W; scale computed inline, not read from HBM).  The
separate rmsnorm kernel is removed.  MoE dispatch: up_gate is per-expert, so the reduction
is recomputed once per (token, expert) copy -> top_k redundant reductions (negligible).
FLOPs intentionally not conserved.
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, standard_tile_points

LABEL = "up_gate_rmsnorm"

REMOVED_GEMM_STAGES = ("up_gate",)
REMOVED_VECTOR_STAGES = ("rmsnorm_square_reduction",)

FP32_BYTES = 4


def kernel_dims(bl) -> tuple[int, int, int]:
    m = bl.padded_m(bl.TOKENS_PER_EXPERT, bl.TENSOR_CORE_MIN_BM)  # per-expert tokens
    n = 2 * bl.INTERMEDIATE_SIZE                                  # gate+up = 4096
    k = bl.HIDDEN_SIZE                                            # 6144 (RMS reduction dim)
    return m, n, k


def _count(bl) -> int:
    return bl.EXPERTS


def tensor_operations(bl) -> float:
    m, n, k = kernel_dims(bl)
    return 2 * m * n * k


def cuda_operations(bl) -> float:
    # Per-expert RMS square-reduction over K (matches ReductionTask op accounting).
    m, _, k = kernel_dims(bl)
    return m * k + m * (k - 1)


def build_frontier(bl):
    m, n, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = standard_tile_points(
        m,
        n,
        k,
        final_tile_bytes=lambda m0, n0: m0 * n0 * b,   # write raw gate+up output
        aux_hbm_bytes=lambda m0, n0, mt, nt: 0,        # RMS scale computed inline; gamma in W
        aux_buffer_bytes=lambda m0, n0: m0 * FP32_BYTES,  # on-chip per-row RMS accumulator
    )
    return build_fused_frontier(
        label=LABEL,
        count=_count(bl),
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
