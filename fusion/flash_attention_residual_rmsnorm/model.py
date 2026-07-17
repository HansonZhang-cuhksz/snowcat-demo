"""Custom fused-kernel model — Fusion 2: MLA output + residual + pre-FFN RMSNorm.

Baseline-agnostic (``bl`` = decode or prefill module).

Fused kernel ``mla_o_residual_rmsnorm``: the ``mla_o`` GEMM epilogue adds the residual,
applies the RMSNorm gamma weight, and accumulates the per-row sum-of-squares (partial
RMS) inline, writing the residual-added/gamma-weighted hidden ``h`` plus a per-row RMS
partial stat.  ``y`` is never written/re-read and ``h`` is never re-read for the
reduction (CODA ``gemm_residual_partial_rms_weight``).
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, standard_tile_points

LABEL = "mla_o_residual_rmsnorm"

REMOVED_GEMM_STAGES = ("mla_o",)
REMOVED_VECTOR_STAGES = ("post_attention_residual_add", "rmsnorm_square_reduction")

FP32_BYTES = 4


def kernel_dims(bl) -> tuple[int, int, int]:
    m = bl.padded_m(bl.BATCH_TOKENS, bl.TENSOR_CORE_MIN_BM)  # tokens (padded)
    n = bl.HIDDEN_SIZE                                       # 6144 (RMS reduction dim)
    k = bl.N_HEADS * bl.V_HEAD_DIM                           # 16384
    return m, n, k


def tensor_operations(bl) -> float:
    m, n, k = kernel_dims(bl)
    return 2 * m * n * k


def cuda_operations(bl) -> float:
    # residual add + RMSNorm square-reduction (both fold into the mla_o epilogue).
    return (
        bl.POST_ATTENTION_RESIDUAL_ADD_TASK.operations
        + bl.RMSNORM_SQUARE_REDUCTION_TASK.operations
    )


def build_frontier(bl):
    m, n, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = standard_tile_points(
        m,
        n,
        k,
        final_tile_bytes=lambda m0, n0: m0 * n0 * b,          # write h = D*gamma
        aux_hbm_bytes=lambda m0, n0, mt, nt: (
            mt * nt * m0 * n0 * b                              # residual C reads
            + mt * nt * n0 * b                                # gamma vector reads
            + mt * nt * m0 * FP32_BYTES                       # partial RMS-stat writes
        ),
        aux_buffer_bytes=lambda m0, n0: (
            m0 * n0 * b                                       # residual tile
            + n0 * b                                          # gamma tile
            + m0 * FP32_BYTES                                 # fp32 partial stat
        ),
    )
    return build_fused_frontier(
        label=LABEL,
        count=1,
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
