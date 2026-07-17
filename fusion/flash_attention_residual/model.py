"""Custom fused-kernel model — Fusion 1: MLA output projection + residual add.

Baseline-agnostic: every entry point takes the baseline module ``bl`` (decode or prefill
``*_area_latency``; identical interface) so the same model serves both stages.

The attention output stage.  Baseline:

    MLA attention core -> mla_o (output projection GEMM) -> y=[TOKENS,HIDDEN] (HBM)
                       -> post_attention_residual_add: out = x_residual + y (HBM)

Fused kernel ``mla_o_residual``: the ``mla_o`` GEMM epilogue adds the residual/skip tensor
to each output tile and writes the sum directly; the raw block output ``y`` is never
written to / re-read from HBM (CODA on-chip intermediate).
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, standard_tile_points

LABEL = "mla_o_residual"

REMOVED_GEMM_STAGES = ("mla_o",)
REMOVED_VECTOR_STAGES = ("post_attention_residual_add",)


def kernel_dims(bl) -> tuple[int, int, int]:
    """(M, N, K) of the fused mla_o GEMM for the currently configured workload."""
    m = bl.padded_m(bl.BATCH_TOKENS, bl.TENSOR_CORE_MIN_BM)  # tokens (padded)
    n = bl.HIDDEN_SIZE                                       # 6144
    k = bl.N_HEADS * bl.V_HEAD_DIM                           # 16384
    return m, n, k


def tensor_operations(bl) -> float:
    m, n, k = kernel_dims(bl)
    return 2 * m * n * k


def cuda_operations(bl) -> float:
    # residual add: one add per [TOKENS, HIDDEN] element (matches the baseline VectorTask).
    return bl.POST_ATTENTION_RESIDUAL_ADD_TASK.operations


def build_frontier(bl):
    m, n, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = standard_tile_points(
        m,
        n,
        k,
        final_tile_bytes=lambda m0, n0: m0 * n0 * b,          # write summed output tile
        aux_hbm_bytes=lambda m0, n0, mt, nt: mt * nt * m0 * n0 * b,  # residual C reads
        aux_buffer_bytes=lambda m0, n0: m0 * n0 * b,          # residual tile on chip
    )
    return build_fused_frontier(
        label=LABEL,
        count=1,
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
