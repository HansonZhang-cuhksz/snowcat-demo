"""Custom fused-kernel model — pre-FFN RMSNorm fused into the router GEMM.

Baseline-agnostic (``bl`` = decode or prefill module).

The router runs on ALL batch tokens (M = batch, N = EXPERTS, K = HIDDEN) and, because MoE
dispatch depends on its logits, it runs *before* up_gate. So the router is a natural place
to compute the pre-FFN RMSNorm scale: it reads ``h`` (the post-residual hidden state) as its
A input anyway, computes the per-row RMS inline (gamma folded into W), applies the scale at
the epilogue, and writes the per-row scale for the downstream up_gate to reuse. The separate
rmsnorm reduction kernel (which re-reads ``h``) is removed.

Crucially, unlike Fusion 3 (RMSNorm+up_gate), the router is M = batch (the unique tokens),
NOT the batch*top_k dispatched tokens — so there is **no ×top_k redundant recompute**; FLOPs
are conserved. And unlike Fusion 2 (RMSNorm into the large compute-bound mla_o), the router
is a small memory-bound GEMM, so its epilogue aux does not starve a big GEMM of SMEM.
"""

from __future__ import annotations

from fusion.common import BYTE_PER_ELEMENT, build_fused_frontier, standard_tile_points

LABEL = "router_rmsnorm"

REMOVED_GEMM_STAGES = ("router",)
REMOVED_VECTOR_STAGES = ("rmsnorm_square_reduction",)

FP32_BYTES = 4


def kernel_dims(bl) -> tuple[int, int, int]:
    m = bl.padded_m(bl.BATCH_TOKENS, bl.TENSOR_CORE_MIN_BM)  # ALL batch tokens (not per-expert)
    n = bl.EXPERTS                                           # 256 routing logits
    k = bl.HIDDEN_SIZE                                       # 6144 (RMS reduction dim)
    return m, n, k


def tensor_operations(bl) -> float:
    m, n, k = kernel_dims(bl)
    return 2 * m * n * k


def cuda_operations(bl) -> float:
    # RMS square-reduction over K, per batch token (M = batch => NO ×top_k redundancy).
    m, _, k = kernel_dims(bl)
    return m * k + m * (k - 1)


def build_frontier(bl):
    m, n, k = kernel_dims(bl)
    b = BYTE_PER_ELEMENT
    points = standard_tile_points(
        m,
        n,
        k,
        final_tile_bytes=lambda m0, n0: m0 * n0 * b,          # write routing logits
        aux_hbm_bytes=lambda m0, n0, mt, nt: mt * m0 * FP32_BYTES,  # write per-row RMS scale
        aux_buffer_bytes=lambda m0, n0: m0 * FP32_BYTES,      # on-chip per-row RMS accumulator
    )
    return build_fused_frontier(
        label=LABEL,
        count=1,
        tensor_operations=tensor_operations(bl),
        cuda_operations=cuda_operations(bl),
        points=points,
        tile_filter=bl.tensor_core_tile_allowed,
    )
