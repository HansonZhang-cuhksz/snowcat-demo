"""Die-area distribution estimator for a full GLM-5.2 *prefill* transformer layer.

The prefill counterpart of ``decode_area_latency.py``.  Prefill processes the entire
prompt (all ``seq_len`` tokens) in one forward pass, so:

    pre-attention RMSNorm
        -> MLA (multi-head latent attention, full causal self-attention; K/V
           materialized per head, NOT the decode matrix-absorbed form)
        -> residual add
        -> pre-FFN RMSNorm
        -> MoE FFN (router, up_gate + SwiGLU, down, expert combine)
        -> post-FFN residual add

Key differences from decode:
- Every GEMM's M is the token count (sequences*seq_len), not the batch of sequences,
  so the projection/FFN GEMMs have huge M and are compute (tensor) bound.
- Attention is quadratic causal self-attention over ``seq_len`` (O(S^2) tensor ops,
  O(S) flash traffic) -> heavily compute-bound, unlike the memory-bound decode KV read.
- The MLA KV path up-projects the latent to per-head K/V (``mla_kv_b``); there is no
  KV cache to stream and no W_UK/W_UV absorption.

Same two estimation tools as the rest of the repo: the Snowcat/Orojenesis Pareto traffic
frontier and the latency-aware roofline with a per-kernel ``num_stages``.

Workload: GLM-5.2 (see memory ``glm-5-2-config``), a single prompt of 1,048,576 tokens
(GLM-5.2 max context), tokens routed evenly across experts.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expert_distribution import ExpertTokenDistribution, binomial_expert_token_distribution
from expert_workload import (
    EvenExpertSplit,
    even_expert_token_split,
    padded_gemm_groups,
    padded_m,
)
from register_accumulator_traffic import (
    FULLY_TILED_REGISTER_ACCUMULATOR_LOOP_ORDERS,
    enumerate_register_accumulator_mappings,
)
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


# Chip constants
# A_total = 694.116 * 10**6              # um^2
A_total = 136.29 * 10**6              # um^2
A_bit = 0.0864                         # um^2/bit
logic_density = 39.98                  # MTr/mm^2 == transistors/um^2
bw = 2.04 * 10**12                     # byte/s

num_sm = 1
A_total = A_total / num_sm
bw = bw / num_sm

# Architecture placeholders. Replace these with measured/spec-sheet values.
CUDA_CORE_TRANSISTORS = 0.2 * 10**6     # transistors/CUDA core placeholder
TENSOR_CORE_TRANSISTORS = 6.0 * 10**6   # transistors/tensor core placeholder
TENSOR_FLOPS = 512 * 1.00 * 10**9       # flops/s/tensor core placeholder
CUDA_CLOCK_HZ = 1410 * 10**6            # Hz
ACTIVATION_FLOPS_PER_CUDA_CORE = 5.64 * 10 ** 9      # flops/s/CUDA core placeholder

A_cuda_core = CUDA_CORE_TRANSISTORS / logic_density
A_tensor_core = TENSOR_CORE_TRANSISTORS / logic_density

# FFN / MoE workload assumptions (GLM-5.2)
BYTE_PER_ELEMENT = 2
EXPERTS = 256
ROUTER_TOP_K = 8
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
TENSOR_CORE_MIN_BM = 16
TENSOR_CORE_MIN_BN = 8
TENSOR_CORE_MIN_BK = 16

# Default workload size.  Override per run via configure() or the CLI
# (--sequences, --seq-len).  Prefill processes all seq_len tokens of every sequence in
# one pass, so the FFN token count is BATCH_TOKENS = PREFILL_SEQUENCES * SEQ_LEN.  Tokens
# route evenly across experts.  All workload-derived globals are (re)bound by configure(),
# which is called at import with these defaults.
DEFAULT_PREFILL_SEQUENCES = 1
DEFAULT_SEQ_LEN = 1_048_576
PREFILL_SEQUENCES = DEFAULT_PREFILL_SEQUENCES
SEQ_LEN = DEFAULT_SEQ_LEN
BATCH_TOKENS = PREFILL_SEQUENCES * SEQ_LEN          # total tokens through the FFN/norms
TOKENS_PER_EXPERT = BATCH_TOKENS * ROUTER_TOP_K // EXPERTS

# Multi-head Latent Attention (MLA) config -- GLM-5.2 (zai-org/GLM-5.2 config.json).
# Prefill materializes per-head K/V from the latent (mla_kv_b up-projection) and runs
# full causal self-attention; there is no KV cache and no W_UK/W_UV absorption.
N_HEADS = 64
KV_LORA_RANK = 512
Q_LORA_RANK = 2048
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 256
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM   # 256, per-head Q/K width
KV_LATENT = KV_LORA_RANK + QK_ROPE_HEAD_DIM         # 576, kv_a output width (c_KV + k_rope)
# per-head K/V produced by the kv_b up-projection: K_nope (qk_nope) + V (v_head); the
# k_rope part (qk_rope) is shared across heads and comes from kv_a.
KV_B_OUT_PER_HEAD = QK_NOPE_HEAD_DIM + V_HEAD_DIM   # 448

# Causal masking: only the lower triangle of the S x S score matrix is computed, ~S^2/2.
CAUSAL_FACTOR = 0.5

# DeepSeek Sparse Attention (DSA) -- GLM-5.2 config.json (zai-org/GLM-5.2).  A lightweight
# "lightning indexer" scores every preceding token (dense O(S^2), but with small
# index heads/dim) and selects the top-``DSA_INDEX_TOPK`` keys per query; the main MLA
# attention then runs only over those selected tokens (O(S * topk)).  Set DSA_ENABLED
# False for dense causal attention.
DSA_ENABLED = True
DSA_INDEX_TOPK = 2048          # selected key tokens per query (GLM-5.2 index_topk)
DSA_INDEX_N_HEADS = 32         # lightning-indexer heads (index_n_heads)
DSA_INDEX_HEAD_DIM = 128       # lightning-indexer head dim (index_head_dim)

# Softmax cost per score element on CUDA cores (exp + online rescale + normalize).
# Scores stay on chip in flash attention, so softmax adds no HBM traffic.
ATTENTION_SOFTMAX_FLOPS_PER_ELEMENT = 5.0

# Nominal flash-attention block (query/key tile rows) held per pipeline stage.  Used only
# to report the attention core's num_stages via the notes' latency formula; the attention
# is compute-bound so the memory term is not on the critical path.
ATTN_FLASH_BLOCK = 128

# Fused SwiGLU activation: SiLU(gate) * up. FLOP accounting for exp/sigmoid is
# implementation dependent, so keep this as a measured/estimated placeholder.
SWIGLU_FLOPS_PER_ELEMENT = 8.0
INCLUDE_RMSNORM = True

CPU_WORKERS = 8
PARALLEL_FRONTIER_MIN_GROUPS = CPU_WORKERS
AREA_GRID_STEP = 0.001

# False preserves the original Snowcat mapspace and output filenames.  True
# restricts GEMM mappings to loop orders that keep output accumulators live
# through their K reduction, closer to real tensor-core GEMM schedules.
USE_REGISTER_ACCUMULATOR_MAPPINGS = True

# False preserves the original even-routing estimate: every expert receives
# TOKENS_PER_EXPERT tokens.  True uses an expected-value random-routing model
# where each expert's token count follows Binomial(BATCH_TOKENS, top_k/EXPERTS).
# MLA is independent of expert routing, so the attention stages are unaffected.
USE_RANDOM_EXPERT_DISTRIBUTION = False
EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF = 1e-12

# Pipeline-depth / latency-hiding model.  num_stages (C) is solved per kernel:
# each concurrent task occupies one Snowcat tile working set (W = buffer_bytes) in
# SMEM, and C tasks stay in flight to hide HBM latency (BW_eff = min(bw, C*W/latency)).
HBM_LATENCY_CYCLES = 500
HBM_CLOCK_HZ = 1215 * 10**6

# True (default) estimates each GEMM's HBM traffic with the Snowcat/Orojenesis
# Pareto frontier (traffic depends on the SMEM-resident tiling) and hides HBM
# latency with the per-tiling num_stages above.  False removes Snowcat: every GEMM
# moves only its algorithmic-minimum HBM traffic (read A and B once, write C once),
# so GEMM OI is independent of the SMEM capacity, and latency hiding treats the
# whole SMEM as one ideal streaming buffer: BW_eff = min(bw, SMEM_total / latency).
# Overly optimistic (assumes perfect on-chip reuse of every operand).  The prefill
# attention core and vector/norm tasks already use analytic operand/result traffic
# and are identical in both modes.
USE_SNOWCAT = True


@dataclass(frozen=True)
class GemmTask:
    name: str
    m: int
    n: int
    k: int

    @property
    def operations(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass(frozen=True)
class GemmTaskGroup:
    label: str
    task: GemmTask
    count: int

    @property
    def operations(self) -> int:
        return self.count * self.task.operations


@dataclass(frozen=True)
class VectorTask:
    name: str
    elements: int
    count: int
    flops_per_element: float
    bytes_per_element_traffic: int

    @property
    def operations(self) -> float:
        return self.count * self.elements * self.flops_per_element

    @property
    def traffic_bytes(self) -> int:
        return self.count * self.elements * self.bytes_per_element_traffic

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class ReductionTask:
    name: str
    rows: int
    columns: int
    bytes_per_input: int
    bytes_per_output: int

    @property
    def operations(self) -> int:
        square_ops = self.rows * self.columns
        reduction_adds = self.rows * (self.columns - 1)
        return square_ops + reduction_adds

    @property
    def traffic_bytes(self) -> int:
        input_reads = self.rows * self.columns * self.bytes_per_input
        output_writes = self.rows * self.bytes_per_output
        return input_reads + output_writes

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class ExpertWeightedSumTask:
    name: str
    tokens: int
    top_k: int
    hidden_size: int
    bytes_per_activation: int
    bytes_per_weight: int
    bytes_per_output: int

    @property
    def operations(self) -> int:
        multiply_ops = self.tokens * self.top_k * self.hidden_size
        add_ops = self.tokens * (self.top_k - 1) * self.hidden_size
        return multiply_ops + add_ops

    @property
    def traffic_bytes(self) -> int:
        activation_reads = (
            self.tokens * self.top_k * self.hidden_size * self.bytes_per_activation
        )
        weight_reads = self.tokens * self.top_k * self.bytes_per_weight
        output_writes = self.tokens * self.hidden_size * self.bytes_per_output
        return activation_reads + weight_reads + output_writes

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class PrefillAttentionTask:
    """Fused flash causal self-attention over the full prompt (prefill), with optional
    DeepSeek Sparse Attention (DSA).

    Dense: per head, scores = Q[S, qk_head] . K^T (causal, ~S^2/2 entries), softmax, then
    AV . V[S, v_head] -- O(S^2) tensor ops, O(S) flash traffic.

    DSA (GLM-5.2): a lightning indexer scores all preceding tokens (dense O(S^2) but with
    ``index_n_heads`` x ``index_head_dim`` small heads) and picks the top-``dsa_topk`` keys
    per query; the main MLA attention then runs only over those selected tokens
    (~S*topk entries per head).  So the quadratic term shrinks to the cheap indexer while
    the main attention becomes linear in S.  Either way it is compute (tensor) bound.
    """

    name: str
    sequences: int
    seq_len: int
    n_heads: int
    qk_head_dim: int
    v_head_dim: int
    causal_factor: float
    softmax_flops_per_element: float
    bytes_per_element: int
    dsa_enabled: bool
    dsa_topk: int
    index_n_heads: int
    index_head_dim: int

    @property
    def dense_score_entries(self) -> float:
        # causal dense (query, key) pairs per head per sequence (~S^2/2)
        return self.seq_len * self.seq_len * self.causal_factor

    @property
    def main_score_entries(self) -> float:
        # (query, key) pairs actually attended by the main MLA attention, per head per seq
        if not self.dsa_enabled:
            return self.dense_score_entries
        k = self.dsa_topk
        s = self.seq_len
        # sum_q min(q+1, topk) over the causal mask = S*topk - topk*(topk-1)/2 (topk < S)
        return min(self.dense_score_entries, s * k - k * (k - 1) / 2.0)

    @property
    def indexer_score_entries(self) -> float:
        # the lightning indexer scores every preceding token: dense O(S^2)
        return self.dense_score_entries if self.dsa_enabled else 0.0

    @property
    def tensor_operations(self) -> float:
        # main MLA attention (QK + AV) over the attended entries
        main = (
            2 * self.n_heads * self.main_score_entries
            * (self.qk_head_dim + self.v_head_dim)
        )
        # lightning indexer scoring (dot product over index_n_heads x index_head_dim)
        indexer = (
            2 * self.index_n_heads * self.index_head_dim * self.indexer_score_entries
        )
        return self.sequences * (main + indexer)

    @property
    def cuda_operations(self) -> float:
        # softmax over the attended entries + indexer ReLU-gating / top-k selection
        softmax = self.softmax_flops_per_element * self.n_heads * self.main_score_entries
        indexer_gate = self.index_n_heads * self.indexer_score_entries
        return self.sequences * (softmax + indexer_gate)

    @property
    def operations(self) -> float:
        return self.tensor_operations + self.cuda_operations

    @property
    def traffic_bytes(self) -> float:
        # Flash attention reads Q, K, V once and writes O (scores stay on chip); DSA adds
        # the indexer key read (index_n_heads x index_head_dim per token).  All O(S).
        b = self.bytes_per_element
        per_seq = (
            self.seq_len * self.n_heads * self.qk_head_dim * b       # Q
            + self.seq_len * self.n_heads * self.qk_head_dim * b     # K (materialized)
            + self.seq_len * self.n_heads * self.v_head_dim * b      # V
            + self.seq_len * self.n_heads * self.v_head_dim * b      # O
        )
        if self.dsa_enabled:
            per_seq += self.seq_len * self.index_n_heads * self.index_head_dim * b
        return self.sequences * per_seq

    @property
    def operational_intensity(self) -> float:
        return self.operations / self.traffic_bytes


@dataclass(frozen=True)
class TrafficFrontier:
    label: str
    count: int
    operations: int
    buffer_bytes: np.ndarray
    traffic_bytes: np.ndarray
    bm: np.ndarray
    bn: np.ndarray
    bk: np.ndarray
    loop_orders: tuple[tuple[str, str, str], ...]


# Workload-derived task objects, (re)bound by configure().  Declared here for clarity;
# configure(DEFAULT_PREFILL_SEQUENCES, DEFAULT_SEQ_LEN) at the bottom of the module populates them.
EVEN_SPLIT: EvenExpertSplit
MLA_GEMM_GROUPS: list[GemmTaskGroup] = []
ATTENTION_CORE_TASK: PrefillAttentionTask
ACTIVATION_TASK: VectorTask
PRE_ATTENTION_RMSNORM_TASK: ReductionTask
RMSNORM_SQUARE_REDUCTION_TASK: ReductionTask
EXPERT_WEIGHTED_SUM_TASK: ExpertWeightedSumTask
POST_ATTENTION_RESIDUAL_ADD_TASK: VectorTask
RESIDUAL_ADD_TASK: VectorTask


def configure(sequences: int, seq_len: int) -> None:
    """(Re)bind PREFILL_SEQUENCES / SEQ_LEN / BATCH_TOKENS / EVEN_SPLIT and every
    workload-derived task object for a given prompt count and length.

    Prefill runs all ``sequences * seq_len`` tokens through the FFN in one pass; tokens
    route evenly across experts via ``even_expert_token_split``.  The downstream
    compute/report functions read these module globals at call time, so calling
    configure() before evaluate_layer()/main() fully reconfigures the workload.
    """
    global PREFILL_SEQUENCES, BATCH_TOKENS, SEQ_LEN, TOKENS_PER_EXPERT, EVEN_SPLIT
    global MLA_GEMM_GROUPS, ATTENTION_CORE_TASK, ACTIVATION_TASK
    global PRE_ATTENTION_RMSNORM_TASK, RMSNORM_SQUARE_REDUCTION_TASK
    global EXPERT_WEIGHTED_SUM_TASK, POST_ATTENTION_RESIDUAL_ADD_TASK, RESIDUAL_ADD_TASK

    if sequences <= 0 or seq_len <= 0:
        raise ValueError("sequences and seq_len must be positive")

    PREFILL_SEQUENCES = sequences
    SEQ_LEN = seq_len
    BATCH_TOKENS = sequences * seq_len          # total prompt tokens through the FFN/norms
    EVEN_SPLIT = even_expert_token_split(BATCH_TOKENS, EXPERTS, ROUTER_TOP_K)
    TOKENS_PER_EXPERT = EVEN_SPLIT.floor_tokens

    # --- MLA projection GEMMs (all prompt tokens live in M) ---
    # GemmTask signature is (name, m, n, k).  Prefill materializes per-head K/V via the
    # kv_b up-projection (no W_UK/W_UV absorption).  M = total tokens (padded to 16).
    token_m = padded_m(BATCH_TOKENS, TENSOR_CORE_MIN_BM)
    MLA_GEMM_GROUPS = [
        GemmTaskGroup(
            "mla_q_a", GemmTask("mla_q_a", token_m, Q_LORA_RANK, HIDDEN_SIZE), 1
        ),
        GemmTaskGroup(
            "mla_q_b",
            GemmTask("mla_q_b", token_m, N_HEADS * QK_HEAD_DIM, Q_LORA_RANK),
            1,
        ),
        GemmTaskGroup(
            "mla_kv_a", GemmTask("mla_kv_a", token_m, KV_LATENT, HIDDEN_SIZE), 1
        ),
        GemmTaskGroup(
            "mla_kv_b",
            GemmTask("mla_kv_b", token_m, N_HEADS * KV_B_OUT_PER_HEAD, KV_LORA_RANK),
            1,
        ),
        GemmTaskGroup(
            "mla_o",
            GemmTask("mla_o", token_m, HIDDEN_SIZE, N_HEADS * V_HEAD_DIM),
            1,
        ),
    ]

    ATTENTION_CORE_TASK = PrefillAttentionTask(
        name="mla_attention",
        sequences=PREFILL_SEQUENCES,
        seq_len=SEQ_LEN,
        n_heads=N_HEADS,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        causal_factor=CAUSAL_FACTOR,
        softmax_flops_per_element=ATTENTION_SOFTMAX_FLOPS_PER_ELEMENT,
        bytes_per_element=BYTE_PER_ELEMENT,
        dsa_enabled=DSA_ENABLED,
        dsa_topk=DSA_INDEX_TOPK,
        index_n_heads=DSA_INDEX_N_HEADS,
        index_head_dim=DSA_INDEX_HEAD_DIM,
    )

    # SwiGLU runs on the real (unpadded) token->expert assignments: sum of tokens over
    # active experts = batch*top_k, independent of the ceil/floor split.
    ACTIVATION_TASK = VectorTask(
        name="activation",
        elements=EVEN_SPLIT.total_assignments * INTERMEDIATE_SIZE,
        count=1,
        flops_per_element=SWIGLU_FLOPS_PER_ELEMENT,
        bytes_per_element_traffic=3 * BYTE_PER_ELEMENT,
    )

    # Pre-attention input RMSNorm (square reduction), batched over BATCH_TOKENS.
    PRE_ATTENTION_RMSNORM_TASK = ReductionTask(
        name="pre_attention_rmsnorm",
        rows=BATCH_TOKENS,
        columns=HIDDEN_SIZE,
        bytes_per_input=BYTE_PER_ELEMENT,
        bytes_per_output=4,
    )

    # Pre-FFN (post-attention) RMSNorm.
    RMSNORM_SQUARE_REDUCTION_TASK = ReductionTask(
        name="rmsnorm_square_reduction",
        rows=BATCH_TOKENS,
        columns=HIDDEN_SIZE,
        bytes_per_input=BYTE_PER_ELEMENT,
        bytes_per_output=4,
    )

    EXPERT_WEIGHTED_SUM_TASK = ExpertWeightedSumTask(
        name="expert_weighted_sum",
        tokens=BATCH_TOKENS,
        top_k=ROUTER_TOP_K,
        hidden_size=HIDDEN_SIZE,
        bytes_per_activation=BYTE_PER_ELEMENT,
        bytes_per_weight=BYTE_PER_ELEMENT,
        bytes_per_output=BYTE_PER_ELEMENT,
    )

    # Residual add after attention.
    POST_ATTENTION_RESIDUAL_ADD_TASK = VectorTask(
        name="post_attention_residual_add",
        elements=BATCH_TOKENS * HIDDEN_SIZE,
        count=1,
        flops_per_element=1.0,
        bytes_per_element_traffic=3 * BYTE_PER_ELEMENT,
    )

    # Residual add after FFN.
    RESIDUAL_ADD_TASK = VectorTask(
        name="residual_add",
        elements=BATCH_TOKENS * HIDDEN_SIZE,
        count=1,
        flops_per_element=1.0,
        bytes_per_element_traffic=3 * BYTE_PER_ELEMENT,
    )


configure(DEFAULT_PREFILL_SEQUENCES, DEFAULT_SEQ_LEN)


def build_even_expert_gemm_groups(
    split: EvenExpertSplit,
) -> tuple[list[GemmTaskGroup], dict[str, float], dict[str, str]]:
    """FFN GEMM groups for even routing.

    Router runs over all EXPERTS.  The per-expert up_gate/down GEMMs are built from the
    even ceil/floor token split: one GEMM group per distinct padded M (M = max(tokens,16)
    for tensor-core feasibility), with ``count`` = number of active experts at that M.
    Groups aggregate under "up_gate"/"down" for reporting.
    """
    router_m = padded_m(BATCH_TOKENS, TENSOR_CORE_MIN_BM)
    task_groups = [
        GemmTaskGroup(
            "router", GemmTask("router", router_m, EXPERTS, HIDDEN_SIZE), 1
        )
    ]
    group_weights = {"router": 1.0}
    aggregate_names = {"router": "router"}

    for task_name, n, k in (
        ("up_gate", 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ("down", HIDDEN_SIZE, INTERMEDIATE_SIZE),
    ):
        m_groups = padded_gemm_groups(split, TENSOR_CORE_MIN_BM)
        multi = len(m_groups) > 1
        for m, count in m_groups:
            label = (
                f"{task_name}_m{m}_x{count}" if multi else f"{task_name}_x{count}"
            )
            task_groups.append(
                GemmTaskGroup(label, GemmTask(task_name, m, n, k), count)
            )
            group_weights[label] = 1.0
            aggregate_names[label] = task_name

    return task_groups, group_weights, aggregate_names


def expert_token_distribution() -> ExpertTokenDistribution:
    return binomial_expert_token_distribution(
        batch_tokens=BATCH_TOKENS,
        experts=EXPERTS,
        top_k=ROUTER_TOP_K,
        probability_cutoff=EXPERT_DISTRIBUTION_PROBABILITY_CUTOFF,
    )


def group_random_expert_gemm_tasks(
    distribution: ExpertTokenDistribution,
) -> tuple[list[GemmTaskGroup], dict[str, float], dict[str, str]]:
    task_groups = [
        GemmTaskGroup(
            label="router",
            task=GemmTask("router", BATCH_TOKENS, EXPERTS, HIDDEN_SIZE),
            count=1,
        )
    ]
    expert_weights = {"router": 1.0}
    aggregate_names = {"router": "router"}

    for task_name, n, k in (
        ("up_gate", 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ("down", HIDDEN_SIZE, INTERMEDIATE_SIZE),
    ):
        aggregate_name = f"{task_name}_random_expected"
        for tokens, probability in distribution.support:
            if tokens == 0:
                continue
            label = f"{task_name}_m{tokens}"
            task_groups.append(
                GemmTaskGroup(
                    label=label,
                    # Pad M to the tensor-core minimum tile when an expert receives
                    # fewer than 16 tokens (tensor core underutilized).
                    task=GemmTask(task_name, padded_m(tokens, TENSOR_CORE_MIN_BM), n, k),
                    count=1,
                )
            )
            expert_weights[label] = EXPERTS * probability
            aggregate_names[label] = aggregate_name

    return task_groups, expert_weights, aggregate_names


def tensor_core_tile_allowed(bm: int, bn: int, bk: int) -> bool:
    return (
        bm >= TENSOR_CORE_MIN_BM
        and bn >= TENSOR_CORE_MIN_BN
        and bk >= TENSOR_CORE_MIN_BK
    )


def algorithmic_min_frontier(group: GemmTaskGroup) -> TrafficFrontier:
    """Single-point frontier carrying the algorithmic-minimum HBM traffic (operands
    read once, result written once) for USE_SNOWCAT=False.  With a 1-byte working
    set the latency model in ``gemm_time_from_frontier`` reduces to
    ``BW_eff = min(bw, SMEM_total / latency)`` -- the whole SMEM acts as one ideal
    streaming buffer -- and the point is valid at any SMEM capacity, so traffic
    (and hence OI) no longer depends on the SMEM budget."""
    task = group.task
    traffic = BYTE_PER_ELEMENT * (
        task.m * task.k + task.k * task.n + task.m * task.n
    )
    return TrafficFrontier(
        label=group.label,
        count=group.count,
        operations=task.operations,
        buffer_bytes=np.array([1], dtype=np.int64),
        traffic_bytes=np.array([traffic], dtype=np.int64),
        bm=np.array([task.m], dtype=np.int64),
        bn=np.array([task.n], dtype=np.int64),
        bk=np.array([task.k], dtype=np.int64),
        loop_orders=(("m", "n", "k"),),
    )


def build_traffic_frontier(group: GemmTaskGroup) -> TrafficFrontier:
    if not USE_SNOWCAT:
        return algorithmic_min_frontier(group)
    task = group.task
    workload = GemmWorkload(
        m=task.m,
        k=task.k,
        n=task.n,
        bytes_per_element=BYTE_PER_ELEMENT,
    )
    mapping_points = (
        enumerate_register_accumulator_mappings(workload)
        if USE_REGISTER_ACCUMULATOR_MAPPINGS
        else enumerate_mappings(workload)
    )
    mapping_points = [
        point
        for point in mapping_points
        if tensor_core_tile_allowed(
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
        )
    ]
    if not mapping_points:
        raise ValueError(
            f"no tensor-core-compatible mapping for {group.label}; "
            f"requires BM>={TENSOR_CORE_MIN_BM}, "
            f"BN>={TENSOR_CORE_MIN_BN}, BK>={TENSOR_CORE_MIN_BK}"
        )
    pairs = sorted(
        (
            point.buffer_bytes,
            point.backing_store_bytes,
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
            point.mapping.loop_order,
        )
        for point in mapping_points
    )

    frontier_buffer_list: list[int] = []
    frontier_traffic_list: list[int] = []
    frontier_bm_list: list[int] = []
    frontier_bn_list: list[int] = []
    frontier_bk_list: list[int] = []
    frontier_loop_order_list: list[tuple[str, str, str]] = []
    best: tuple[int, int, int, int, tuple[str, str, str]] | None = None

    for buffer_bytes, traffic_bytes, bm, bn, bk, loop_order in pairs:
        if best is None or traffic_bytes < best[0]:
            best = (traffic_bytes, bm, bn, bk, loop_order)
        if frontier_buffer_list and buffer_bytes == frontier_buffer_list[-1]:
            frontier_traffic_list[-1] = best[0]
            frontier_bm_list[-1] = best[1]
            frontier_bn_list[-1] = best[2]
            frontier_bk_list[-1] = best[3]
            frontier_loop_order_list[-1] = best[4]
        else:
            frontier_buffer_list.append(buffer_bytes)
            frontier_traffic_list.append(best[0])
            frontier_bm_list.append(best[1])
            frontier_bn_list.append(best[2])
            frontier_bk_list.append(best[3])
            frontier_loop_order_list.append(best[4])

    frontier_buffers = np.array(frontier_buffer_list, dtype=np.int64)
    frontier_traffic = np.array(frontier_traffic_list, dtype=np.int64)
    frontier_bm = np.array(frontier_bm_list, dtype=np.int64)
    frontier_bn = np.array(frontier_bn_list, dtype=np.int64)
    frontier_bk = np.array(frontier_bk_list, dtype=np.int64)
    improved = np.r_[True, frontier_traffic[1:] < frontier_traffic[:-1]]

    return TrafficFrontier(
        label=group.label,
        count=group.count,
        operations=group.task.operations,
        buffer_bytes=frontier_buffers[improved],
        traffic_bytes=frontier_traffic[improved],
        bm=frontier_bm[improved],
        bn=frontier_bn[improved],
        bk=frontier_bk[improved],
        loop_orders=tuple(
            loop_order
            for keep, loop_order in zip(improved, frontier_loop_order_list)
            if keep
        ),
    )


def output_paths() -> tuple[str, str, str]:
    suffix_parts = []
    if USE_REGISTER_ACCUMULATOR_MAPPINGS:
        suffix_parts.append("register_accumulator")
    if USE_RANDOM_EXPERT_DISTRIBUTION:
        suffix_parts.append("random_experts")
    if not USE_SNOWCAT:
        suffix_parts.append("no_snowcat")
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    return (
        f"./result/prefill_area_latency{suffix}_times.csv",
        f"./result/prefill_area_latency{suffix}_total_time.png",
        f"./result/prefill_area_latency{suffix}_attention_time.png",
    )


def build_frontiers(task_groups: list[GemmTaskGroup]) -> tuple[list[TrafficFrontier], str]:
    if len(task_groups) < PARALLEL_FRONTIER_MIN_GROUPS:
        return [build_traffic_frontier(group) for group in task_groups], "serial"

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
        frontiers = list(
            executor.map(build_traffic_frontier, task_groups, chunksize=1)
        )
    return frontiers, f"{CPU_WORKERS}-process"


def gemm_time_from_frontier(
    frontier: TrafficFrontier, s_total: np.ndarray, tensor_roof: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-area-node stage time and HBM traffic, minimized over Pareto points.

    For each frontier point (tiling) ``i`` with one-stage working set ``W_i`` and
    HBM traffic ``T_i``, the optimal pipeline depth is
    ``C_best = min(floor(S_total / W_i), ceil(bw * latency / W_i))`` (smallest
    ``num_stages`` achieving that tiling's minimum time; ``ceil(bw*latency/W_i)``
    is where physical BW saturates).  ``BW_eff = min(bw, C_best * W_i / latency)``
    and the stage time is ``count * max(ops / tensor_roof, T_i / BW_eff)``.  The
    per-node minimum over points is returned, together with the winning point's
    traffic.
    """
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    n = len(s_total)
    time_best = np.full(n, np.inf, dtype=float)
    traffic_best = np.full(n, np.nan, dtype=float)
    tensor_time = np.full(n, np.inf, dtype=float)
    np.divide(frontier.operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    for i in range(len(frontier.buffer_bytes)):
        w_i = float(frontier.buffer_bytes[i])
        t_i = float(frontier.traffic_bytes[i])
        c_max = np.floor(s_total / w_i)
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * latency_seconds / w_i))
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w_i / latency_seconds)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem_time = t_i / bw_eff
        time_i = frontier.count * np.maximum(tensor_time, mem_time)
        time_i = np.where(valid, time_i, np.inf)
        better = time_i < time_best
        time_best = np.where(better, time_i, time_best)
        traffic_best = np.where(better, t_i, traffic_best)
    return time_best, traffic_best


def select_mapping_from_frontier(
    frontier: TrafficFrontier, s_total: float, tensor_roof: float
) -> dict[str, object] | None:
    """Winning tiling + num_stages at a single (fixed) SMEM budget."""
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    best: dict[str, object] | None = None
    tensor_time = (
        frontier.operations / tensor_roof if tensor_roof > 0 else np.inf
    )
    for i in range(len(frontier.buffer_bytes)):
        w_i = int(frontier.buffer_bytes[i])
        t_i = int(frontier.traffic_bytes[i])
        c_max = int(s_total // w_i)
        if c_max < 1:
            continue
        c_sat = int(np.ceil(bw * latency_seconds / w_i))
        c_best = min(c_max, c_sat)
        bw_eff = min(bw, c_best * w_i / latency_seconds)
        mem_time = t_i / bw_eff
        time_i = frontier.count * max(tensor_time, mem_time)
        if best is None or time_i < best["time"]:  # type: ignore[typeddict-item]
            best = {
                "bm": int(frontier.bm[i]),
                "bn": int(frontier.bn[i]),
                "bk": int(frontier.bk[i]),
                "loop_order": frontier.loop_orders[i],
                "num_stages": c_best,
                "max_feasible_stages": c_max,
                "one_stage_smem": w_i,
                "traffic": t_i,
                "oi": frontier.operations / t_i,
                "bw_eff": bw_eff,
                "time": time_i,
            }
    return best


def format_selected_mapping(
    frontier: TrafficFrontier, s_total: float, tensor_roof: float
) -> str:
    mapping = select_mapping_from_frontier(frontier, s_total, tensor_roof)
    if mapping is None:
        return "no mapping fits selected SMEM capacity"
    if not USE_SNOWCAT:
        return (
            "algorithmic-min traffic (Snowcat disabled): "
            f"traffic={mapping['traffic'] / 2**20:.3f} MiB, "
            f"OI={mapping['oi']:.6f} FLOP/byte, "
            f"BW_eff={mapping['bw_eff'] / 1e12:.6f} TB/s "
            "(whole-SMEM streaming buffer)"
        )
    return (
        f"BM={mapping['bm']}, BN={mapping['bn']}, BK={mapping['bk']}, "
        f"loop_order={'-'.join(mapping['loop_order'])}, "
        f"num_stages={mapping['num_stages']} "
        f"(max_feasible={mapping['max_feasible_stages']}), "
        f"one_stage_smem={mapping['one_stage_smem'] / 2**20:.6f} MiB "
        f"({mapping['one_stage_smem']} bytes), "
        f"traffic={mapping['traffic'] / 2**20:.3f} MiB, "
        f"OI={mapping['oi']:.6f} FLOP/byte, "
        f"BW_eff={mapping['bw_eff'] / 1e12:.6f} TB/s"
    )


def _attn_stage_bytes(task: PrefillAttentionTask) -> float:
    """One flash-attention pipeline stage: a Q/K/V tile of ATTN_FLASH_BLOCK rows across
    all heads (nominal, for num_stages reporting only)."""
    return float(
        ATTN_FLASH_BLOCK
        * task.n_heads
        * (task.qk_head_dim + task.v_head_dim)
        * task.bytes_per_element
    )


def attention_core_time(
    task: PrefillAttentionTask,
    s_total: np.ndarray,
    tensor_roof: np.ndarray,
    cuda_roof: np.ndarray,
) -> np.ndarray:
    """Latency-aware time of the fused causal flash-attention core per area node.

    Roofline over the three fused bottlenecks (mirrors the fused-GEMM stage form):
    ``time = max(tensor_ops/tensor_roof, softmax_ops/cuda_roof, traffic/BW_eff)``.
    Prefill attention is O(S^2) compute vs O(S) flash traffic, so the tensor term
    dominates; the latency-aware BW_eff (nominal per-stage flash tile ``W_stage``) is
    reported for completeness but is not on the critical path.
    """
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    w_stage = _attn_stage_bytes(task)
    c_max = np.floor(s_total / w_stage)
    valid = c_max >= 1
    c_sat = int(np.ceil(bw * latency_seconds / w_stage))
    c_best = np.minimum(c_max, c_sat)
    c_safe = np.where(valid, c_best, 1.0)
    bw_eff = np.minimum(bw, c_safe * w_stage / latency_seconds)

    tensor_time = np.full_like(s_total, np.inf, dtype=float)
    np.divide(task.tensor_operations, tensor_roof, out=tensor_time, where=tensor_roof > 0)
    cuda_time = np.full_like(s_total, np.inf, dtype=float)
    np.divide(task.cuda_operations, cuda_roof, out=cuda_time, where=cuda_roof > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mem_time = task.traffic_bytes / bw_eff

    time_seconds = np.maximum(np.maximum(tensor_time, cuda_time), mem_time)
    return np.where(valid, time_seconds, np.inf)


def attention_core_mapping(
    task: PrefillAttentionTask,
    s_total: float,
    tensor_roof: float,
    cuda_roof: float,
) -> dict[str, object]:
    """Reporting helper: num_stages / BW_eff / OI / bottleneck at one SMEM budget."""
    latency_seconds = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ
    w_stage = _attn_stage_bytes(task)
    c_max = int(s_total // w_stage)
    c_sat = int(np.ceil(bw * latency_seconds / w_stage))
    c_best = max(min(c_max, c_sat), 1)
    bw_eff = min(bw, c_best * w_stage / latency_seconds)

    tensor_time = task.tensor_operations / tensor_roof if tensor_roof > 0 else np.inf
    cuda_time = task.cuda_operations / cuda_roof if cuda_roof > 0 else np.inf
    mem_time = task.traffic_bytes / bw_eff
    time_seconds = max(tensor_time, cuda_time, mem_time)
    if time_seconds == mem_time:
        bottleneck = "memory"
    elif time_seconds == tensor_time:
        bottleneck = "tensor"
    else:
        bottleneck = "cuda"

    return {
        "flash_block_rows": ATTN_FLASH_BLOCK,
        "num_stages": c_best,
        "max_feasible_stages": c_max,
        "one_stage_smem": int(w_stage),
        "traffic": task.traffic_bytes,
        "oi": task.operational_intensity,
        "bw_eff": bw_eff,
        "tensor_time": tensor_time,
        "cuda_time": cuda_time,
        "mem_time": mem_time,
        "time": time_seconds,
        "bottleneck": bottleneck,
    }


def format_attention_mapping(
    task: PrefillAttentionTask, s_total: float, tensor_roof: float, cuda_roof: float
) -> str:
    m = attention_core_mapping(task, s_total, tensor_roof, cuda_roof)
    return (
        f"flash_block={m['flash_block_rows']} rows, "
        f"num_stages={m['num_stages']} (max_feasible={m['max_feasible_stages']}), "
        f"one_stage_smem={m['one_stage_smem'] / 2**10:.3f} KiB, "
        f"OI={m['oi']:.6f} FLOP/byte, "
        f"BW_eff={m['bw_eff'] / 1e12:.6f} TB/s, "
        f"bottleneck={m['bottleneck']} "
        f"(tensor={m['tensor_time'] * 1e3:.3f} ms, "
        f"cuda={m['cuda_time'] * 1e3:.3f} ms, "
        f"mem={m['mem_time'] * 1e3:.3f} ms)"
    )


def vector_time(task: VectorTask, cuda_roof: np.ndarray) -> np.ndarray:
    memory_roof = task.operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(task.operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def reduction_time(task: ReductionTask, cuda_roof: np.ndarray) -> np.ndarray:
    memory_roof = task.operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(task.operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def streaming_cuda_time(
    operations: int | float, traffic_bytes: int | float, cuda_roof: np.ndarray
) -> np.ndarray:
    operational_intensity = operations / traffic_bytes
    memory_roof = operational_intensity * bw
    peak = np.minimum(memory_roof, cuda_roof)

    time_seconds = np.full(len(cuda_roof), np.nan, dtype=float)
    np.divide(operations, peak, out=time_seconds, where=peak > 0)
    return time_seconds


def total_hbm_traffic_bytes(task_traffic: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros_like(next(iter(task_traffic.values())), dtype=float)
    for traffic_bytes in task_traffic.values():
        total = total + traffic_bytes
    total = total + PRE_ATTENTION_RMSNORM_TASK.traffic_bytes
    total = total + ATTENTION_CORE_TASK.traffic_bytes
    total = total + POST_ATTENTION_RESIDUAL_ADD_TASK.traffic_bytes
    if INCLUDE_RMSNORM:
        total = total + RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes
    total = total + ACTIVATION_TASK.traffic_bytes
    total = total + EXPERT_WEIGHTED_SUM_TASK.traffic_bytes
    total = total + RESIDUAL_ADD_TASK.traffic_bytes
    return total


def make_area_grid(step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.arange(step, 1.0, step)
    rc_values = []
    rt_values = []
    smem_values = []

    for rc in values:
        for rt in values:
            smem = 1.0 - rc - rt
            if smem > 0.0:
                rc_values.append(rc)
                rt_values.append(rt)
                smem_values.append(smem)

    return (
        np.array(rc_values, dtype=float),
        np.array(rt_values, dtype=float),
        np.array(smem_values, dtype=float),
    )


def write_csv(
    path: str,
    rc: np.ndarray,
    rt: np.ndarray,
    r_smem: np.ndarray,
    smem_bytes: np.ndarray,
    cuda_cores: np.ndarray,
    tensor_cores: np.ndarray,
    task_times: dict[str, np.ndarray],
    task_traffic: dict[str, np.ndarray],
    pre_attention_rmsnorm_time: np.ndarray,
    attention_time: np.ndarray,
    post_attention_residual_add_time: np.ndarray,
    rmsnorm_time: np.ndarray,
    activation_time: np.ndarray,
    expert_weighted_sum_time: np.ndarray,
    residual_add_time: np.ndarray,
    total_time: np.ndarray,
    modeled_operations: float,
) -> None:
    with open(path, "w", newline="") as csvfile:
        fieldnames = [
            "rc",
            "rt",
            "r_smem",
            "smem_mib",
            "cuda_cores",
            "tensor_cores",
            "total_time_ms",
            "total_hbm_mib",
            "effective_tflops",
            "pre_attention_rmsnorm_oi_flops_per_byte",
            "mla_attention_oi_flops_per_byte",
            "post_attention_residual_add_oi_flops_per_byte",
            "activation_oi_flops_per_byte",
            "rmsnorm_square_reduction_oi_flops_per_byte",
            "expert_weighted_sum_oi_flops_per_byte",
            "residual_add_oi_flops_per_byte",
            *[f"{name}_time_ms" for name in task_times],
            "pre_attention_rmsnorm_time_ms",
            "mla_attention_time_ms",
            "post_attention_residual_add_time_ms",
            "rmsnorm_square_reduction_time_ms",
            "activation_time_ms",
            "expert_weighted_sum_time_ms",
            "residual_add_time_ms",
            *[f"{name}_hbm_mib" for name in task_traffic],
            "pre_attention_rmsnorm_hbm_mib",
            "mla_attention_hbm_mib",
            "post_attention_residual_add_hbm_mib",
            "rmsnorm_square_reduction_hbm_mib",
            "activation_hbm_mib",
            "expert_weighted_sum_hbm_mib",
            "residual_add_hbm_mib",
            *[f"{name}_oi_flops_per_byte" for name in task_traffic],
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        total_hbm_traffic = total_hbm_traffic_bytes(task_traffic)

        for index in range(len(rc)):
            row = {
                "rc": rc[index],
                "rt": rt[index],
                "r_smem": r_smem[index],
                "smem_mib": smem_bytes[index] / 2**20,
                "cuda_cores": cuda_cores[index],
                "tensor_cores": tensor_cores[index],
                "total_time_ms": total_time[index] * 1e3,
                "total_hbm_mib": total_hbm_traffic[index] / 2**20,
                "effective_tflops": modeled_operations / total_time[index] / 1e12,
                "pre_attention_rmsnorm_oi_flops_per_byte": (
                    PRE_ATTENTION_RMSNORM_TASK.operational_intensity
                ),
                "mla_attention_oi_flops_per_byte": (
                    ATTENTION_CORE_TASK.operational_intensity
                ),
                "post_attention_residual_add_oi_flops_per_byte": (
                    POST_ATTENTION_RESIDUAL_ADD_TASK.operational_intensity
                ),
                "activation_oi_flops_per_byte": ACTIVATION_TASK.operational_intensity,
                "rmsnorm_square_reduction_oi_flops_per_byte": (
                    RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity
                ),
                "expert_weighted_sum_oi_flops_per_byte": (
                    EXPERT_WEIGHTED_SUM_TASK.operational_intensity
                ),
                "residual_add_oi_flops_per_byte": (
                    RESIDUAL_ADD_TASK.operational_intensity
                ),
            }
            for name, time_seconds in task_times.items():
                row[f"{name}_time_ms"] = time_seconds[index] * 1e3
            row["pre_attention_rmsnorm_time_ms"] = (
                pre_attention_rmsnorm_time[index] * 1e3
            )
            row["mla_attention_time_ms"] = attention_time[index] * 1e3
            row["post_attention_residual_add_time_ms"] = (
                post_attention_residual_add_time[index] * 1e3
            )
            row["rmsnorm_square_reduction_time_ms"] = rmsnorm_time[index] * 1e3
            row["activation_time_ms"] = activation_time[index] * 1e3
            row["expert_weighted_sum_time_ms"] = (
                expert_weighted_sum_time[index] * 1e3
            )
            row["residual_add_time_ms"] = residual_add_time[index] * 1e3
            for name, traffic_bytes in task_traffic.items():
                row[f"{name}_hbm_mib"] = traffic_bytes[index] / 2**20
            row["pre_attention_rmsnorm_hbm_mib"] = (
                PRE_ATTENTION_RMSNORM_TASK.traffic_bytes / 2**20
            )
            row["mla_attention_hbm_mib"] = ATTENTION_CORE_TASK.traffic_bytes / 2**20
            row["post_attention_residual_add_hbm_mib"] = (
                POST_ATTENTION_RESIDUAL_ADD_TASK.traffic_bytes / 2**20
            )
            row["rmsnorm_square_reduction_hbm_mib"] = (
                RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20
            )
            row["activation_hbm_mib"] = ACTIVATION_TASK.traffic_bytes / 2**20
            row["expert_weighted_sum_hbm_mib"] = (
                EXPERT_WEIGHTED_SUM_TASK.traffic_bytes / 2**20
            )
            row["residual_add_hbm_mib"] = RESIDUAL_ADD_TASK.traffic_bytes / 2**20
            for name, traffic_bytes in task_traffic.items():
                row[f"{name}_oi_flops_per_byte"] = (
                    task_operations_by_name[name] / traffic_bytes[index]
                )
            writer.writerow(row)


def plot_results(
    rc: np.ndarray,
    rt: np.ndarray,
    total_time: np.ndarray,
    attention_time: np.ndarray,
    total_time_path: str,
    attention_time_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Path("result").mkdir(exist_ok=True)
    valid = np.isfinite(total_time) & (total_time > 0) & (attention_time > 0)

    plt.figure(figsize=(10, 7))
    total_time_ms = total_time[valid] * 1e3
    scatter = plt.scatter(
        rt[valid],
        rc[valid],
        c=total_time_ms,
        s=8,
        cmap="viridis_r",
        norm=LogNorm(vmin=total_time_ms.min(), vmax=total_time_ms.max()),
    )
    plt.colorbar(scatter, label="Total time (ms)")
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title("GLM-5.2 Prefill Layer Time Across CUDA/Tensor/SMEM Area Split")
    plt.tight_layout()
    plt.savefig(total_time_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    attention_time_ms = attention_time[valid] * 1e3
    scatter = plt.scatter(
        rt[valid],
        rc[valid],
        c=attention_time_ms,
        s=8,
        cmap="magma_r",
        norm=LogNorm(vmin=attention_time_ms.min(), vmax=attention_time_ms.max()),
    )
    plt.colorbar(scatter, label="MLA attention core time (ms)")
    plt.xlabel("Tensor-core area fraction rt")
    plt.ylabel("CUDA-core area fraction rc")
    plt.title("MLA Causal Attention Core Time Across Area Split")
    plt.tight_layout()
    plt.savefig(attention_time_path, dpi=160)
    plt.close()


task_operations_by_name: dict[str, float] = {}


def evaluate_layer() -> dict[str, object]:
    """Compute per-area-node stage times/traffic and the best area node for the
    currently configured workload (see ``configure()``).

    Pure compute -- no file output -- so it can be swept in-process across batch
    sizes / sequence lengths.  Returns every array and bookkeeping structure the
    reporting/writing in ``main()`` needs, plus ``best_index`` (argmin total time).
    """
    task_operations_by_name.clear()

    rc, rt, r_smem = make_area_grid(AREA_GRID_STEP)
    smem_bytes = r_smem * A_total / A_bit / 8
    cuda_cores = np.floor(rc * A_total / A_cuda_core)
    tensor_cores = np.floor(rt * A_total / A_tensor_core)
    cuda_roof = cuda_cores * ACTIVATION_FLOPS_PER_CUDA_CORE
    tensor_roof = tensor_cores * TENSOR_FLOPS
    flops_per_cuda_core_cycle = ACTIVATION_FLOPS_PER_CUDA_CORE / CUDA_CLOCK_HZ

    distribution = (
        expert_token_distribution() if USE_RANDOM_EXPERT_DISTRIBUTION else None
    )
    if distribution is None:
        ffn_groups, group_weights, aggregate_names = build_even_expert_gemm_groups(
            EVEN_SPLIT
        )
    else:
        ffn_groups, group_weights, aggregate_names = group_random_expert_gemm_tasks(
            distribution
        )

    # MLA projection/absorption GEMMs are independent of expert routing; prepend
    # them so the attention block precedes the FFN block in the reported order.
    for group in MLA_GEMM_GROUPS:
        group_weights[group.label] = 1.0
        aggregate_names[group.label] = group.label
    task_groups = MLA_GEMM_GROUPS + ffn_groups

    frontiers, frontier_mode = build_frontiers(task_groups)
    gemm_operations = 0.0
    task_times: dict[str, np.ndarray] = {}
    task_traffic: dict[str, np.ndarray] = {}
    for frontier in frontiers:
        aggregate_name = aggregate_names[frontier.label]
        weight = group_weights[frontier.label]
        weighted_operations = weight * frontier.count * frontier.operations
        gemm_operations += weighted_operations
        task_operations_by_name[aggregate_name] = (
            task_operations_by_name.get(aggregate_name, 0.0) + weighted_operations
        )

        time_seconds, traffic_bytes = gemm_time_from_frontier(
            frontier, smem_bytes, tensor_roof
        )
        weighted_time = weight * time_seconds
        weighted_traffic = weight * frontier.count * traffic_bytes
        if aggregate_name in task_times:
            task_times[aggregate_name] = task_times[aggregate_name] + weighted_time
            task_traffic[aggregate_name] = (
                task_traffic[aggregate_name] + weighted_traffic
            )
        else:
            task_times[aggregate_name] = weighted_time
            task_traffic[aggregate_name] = weighted_traffic

    rmsnorm_operations = (
        RMSNORM_SQUARE_REDUCTION_TASK.operations if INCLUDE_RMSNORM else 0
    )
    modeled_operations = (
        gemm_operations
        + PRE_ATTENTION_RMSNORM_TASK.operations
        + ATTENTION_CORE_TASK.operations
        + POST_ATTENTION_RESIDUAL_ADD_TASK.operations
        + ACTIVATION_TASK.operations
        + rmsnorm_operations
        + EXPERT_WEIGHTED_SUM_TASK.operations
        + RESIDUAL_ADD_TASK.operations
    )

    pre_attention_rmsnorm_time = reduction_time(
        PRE_ATTENTION_RMSNORM_TASK, cuda_roof
    )
    attention_time = attention_core_time(
        ATTENTION_CORE_TASK, smem_bytes, tensor_roof, cuda_roof
    )
    post_attention_residual_add_time = vector_time(
        POST_ATTENTION_RESIDUAL_ADD_TASK, cuda_roof
    )
    rmsnorm_time = (
        reduction_time(RMSNORM_SQUARE_REDUCTION_TASK, cuda_roof)
        if INCLUDE_RMSNORM
        else np.zeros(len(rc), dtype=float)
    )
    activation_time = vector_time(ACTIVATION_TASK, cuda_roof)
    expert_weighted_sum_time = streaming_cuda_time(
        EXPERT_WEIGHTED_SUM_TASK.operations,
        EXPERT_WEIGHTED_SUM_TASK.traffic_bytes,
        cuda_roof,
    )
    residual_add_time = vector_time(RESIDUAL_ADD_TASK, cuda_roof)
    total_time = np.sum(
        np.array(
            [
                *task_times.values(),
                pre_attention_rmsnorm_time,
                attention_time,
                post_attention_residual_add_time,
                rmsnorm_time,
                activation_time,
                expert_weighted_sum_time,
                residual_add_time,
            ]
        ),
        axis=0,
    )

    best_index = int(np.nanargmin(total_time))

    return {
        "rc": rc,
        "rt": rt,
        "r_smem": r_smem,
        "smem_bytes": smem_bytes,
        "cuda_cores": cuda_cores,
        "tensor_cores": tensor_cores,
        "cuda_roof": cuda_roof,
        "tensor_roof": tensor_roof,
        "flops_per_cuda_core_cycle": flops_per_cuda_core_cycle,
        "distribution": distribution,
        "ffn_groups": ffn_groups,
        "task_groups": task_groups,
        "frontiers": frontiers,
        "frontier_mode": frontier_mode,
        "group_weights": group_weights,
        "aggregate_names": aggregate_names,
        "gemm_operations": gemm_operations,
        "task_times": task_times,
        "task_traffic": task_traffic,
        "pre_attention_rmsnorm_time": pre_attention_rmsnorm_time,
        "attention_time": attention_time,
        "post_attention_residual_add_time": post_attention_residual_add_time,
        "rmsnorm_time": rmsnorm_time,
        "activation_time": activation_time,
        "expert_weighted_sum_time": expert_weighted_sum_time,
        "residual_add_time": residual_add_time,
        "total_time": total_time,
        "modeled_operations": modeled_operations,
        "best_index": best_index,
    }


def main(write_outputs: bool = True) -> dict[str, object]:
    results = evaluate_layer()
    rc = results["rc"]
    rt = results["rt"]
    r_smem = results["r_smem"]
    smem_bytes = results["smem_bytes"]
    cuda_cores = results["cuda_cores"]
    tensor_cores = results["tensor_cores"]
    cuda_roof = results["cuda_roof"]
    tensor_roof = results["tensor_roof"]
    flops_per_cuda_core_cycle = results["flops_per_cuda_core_cycle"]
    distribution = results["distribution"]
    ffn_groups = results["ffn_groups"]
    task_groups = results["task_groups"]
    frontiers = results["frontiers"]
    frontier_mode = results["frontier_mode"]
    group_weights = results["group_weights"]
    aggregate_names = results["aggregate_names"]
    task_times = results["task_times"]
    task_traffic = results["task_traffic"]
    pre_attention_rmsnorm_time = results["pre_attention_rmsnorm_time"]
    attention_time = results["attention_time"]
    post_attention_residual_add_time = results["post_attention_residual_add_time"]
    rmsnorm_time = results["rmsnorm_time"]
    activation_time = results["activation_time"]
    expert_weighted_sum_time = results["expert_weighted_sum_time"]
    residual_add_time = results["residual_add_time"]
    total_time = results["total_time"]
    modeled_operations = results["modeled_operations"]
    best_index = results["best_index"]

    csv_path, total_time_plot_path, attention_time_plot_path = output_paths()
    if write_outputs:
        Path("result").mkdir(exist_ok=True)
        write_csv(
            csv_path,
            rc,
            rt,
            r_smem,
            smem_bytes,
            cuda_cores,
            tensor_cores,
            task_times,
            task_traffic,
            pre_attention_rmsnorm_time,
            attention_time,
            post_attention_residual_add_time,
            rmsnorm_time,
            activation_time,
            expert_weighted_sum_time,
            residual_add_time,
            total_time,
            modeled_operations,
        )
        plot_results(
            rc,
            rt,
            total_time,
            attention_time,
            total_time_plot_path,
            attention_time_plot_path,
        )

    effective_flops = modeled_operations / total_time[best_index]
    total_hbm_traffic = total_hbm_traffic_bytes(task_traffic)

    print("\n=== Configuration ===")
    print(f"CPU workers available: {CPU_WORKERS}")
    print(f"Frontier build mode: {frontier_mode}")
    print(
        "Traffic model: "
        + (
            (
                "register-accumulator loop orders only"
                if USE_REGISTER_ACCUMULATOR_MAPPINGS
                else "original Snowcat all-loop-order mapspace"
            )
            if USE_SNOWCAT
            else "NO SNOWCAT -- algorithmic-minimum HBM traffic "
            "(operands + results only; OI independent of SMEM; "
            "BW_eff = min(bw, SMEM/latency))"
        )
    )
    if USE_SNOWCAT and USE_REGISTER_ACCUMULATOR_MAPPINGS:
        allowed = [
            "-".join(loop_order)
            for loop_order in FULLY_TILED_REGISTER_ACCUMULATOR_LOOP_ORDERS
        ]
        print(f"Fully tiled allowed loop orders: {', '.join(allowed)}")
    print(
        f"Output CSV: {csv_path}"
        + ("" if write_outputs else "  (not written; --no-write)")
    )
    print(f"Prefill sequences: {PREFILL_SEQUENCES}")
    print(f"Sequence length (prompt): {SEQ_LEN}")
    print(f"Total prompt tokens (FFN batch): {BATCH_TOKENS}")
    print(f"Router top-k: {ROUTER_TOP_K}")
    print(f"Expert token split: {EVEN_SPLIT.summary()}")
    if EVEN_SPLIT.ceil_tokens < TENSOR_CORE_MIN_BM:
        print(
            f"  per-expert GEMM M padded to {TENSOR_CORE_MIN_BM} "
            f"(tensor core underutilized; real tokens/expert < {TENSOR_CORE_MIN_BM})"
        )
    print(f"Unique GEMM groups: {len(task_groups)} (MLA {len(MLA_GEMM_GROUPS)} + FFN {len(ffn_groups)})")
    print(f"HBM latency: {HBM_LATENCY_CYCLES} cycles")
    print(
        "Expert distribution model: "
        + (
            "random binomial expected value"
            if USE_RANDOM_EXPERT_DISTRIBUTION
            else "even deterministic tokens per expert"
        )
    )
    if distribution is not None:
        support_counts = [tokens for tokens, _ in distribution.support]
        print(
            "Per-expert token count: "
            f"Binomial(n={distribution.batch_tokens}, "
            f"p={distribution.selection_probability:.8f})"
        )
        print(
            "Per-expert token mean/stddev: "
            f"{distribution.mean:.6f}/"
            f"{np.sqrt(distribution.variance):.6f}"
        )
        print(
            "Retained token-count support: "
            f"{min(support_counts)}..{max(support_counts)} "
            f"({len(support_counts)} counts), "
            f"probability mass {distribution.retained_probability_mass:.12f}"
        )
    print("\n--- MLA (multi-head latent attention, prefill causal self-attention) ---")
    print(f"Attention heads: {N_HEADS}")
    print(
        f"kv_lora_rank={KV_LORA_RANK}, q_lora_rank={Q_LORA_RANK}, "
        f"qk_nope_head_dim={QK_NOPE_HEAD_DIM}, qk_rope_head_dim={QK_ROPE_HEAD_DIM}, "
        f"v_head_dim={V_HEAD_DIM}"
    )
    print(
        f"Per-head Q/K width={QK_HEAD_DIM}, kv_b out per head={KV_B_OUT_PER_HEAD} "
        f"(K_nope {QK_NOPE_HEAD_DIM} + V {V_HEAD_DIM})"
    )
    if DSA_ENABLED:
        print(
            f"DeepSeek Sparse Attention: top-{DSA_INDEX_TOPK} keys/query "
            f"(indexer {DSA_INDEX_N_HEADS} heads x {DSA_INDEX_HEAD_DIM} dim), causal factor {CAUSAL_FACTOR}"
        )
    else:
        print(f"Dense causal attention (causal factor {CAUSAL_FACTOR})")
    print(f"Attention core OI: {ATTENTION_CORE_TASK.operational_intensity:.3f} FLOP/byte (compute-bound)")
    print(f"CUDA FLOP/cycle/core: {flops_per_cuda_core_cycle:.6f}")
    print(f"RMSNorm square-reduction enabled: {INCLUDE_RMSNORM}")
    print(
        "Pre-attention RMSNorm OI: "
        f"{PRE_ATTENTION_RMSNORM_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(
        "RMSNorm square-reduction OI: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Activation OI: {ACTIVATION_TASK.operational_intensity:.6f} FLOP/byte")
    print(
        "Expert weighted-sum OI: "
        f"{EXPERT_WEIGHTED_SUM_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(
        "Post-attention residual add OI: "
        f"{POST_ATTENTION_RESIDUAL_ADD_TASK.operational_intensity:.6f} FLOP/byte"
    )
    print(f"Residual add OI: {RESIDUAL_ADD_TASK.operational_intensity:.6f} FLOP/byte")

    print("\n=== Best Area Point ===")
    print(f"Best rc: {rc[best_index]:.6g}")
    print(f"Best rt: {rt[best_index]:.6g}")
    print(f"Best SMEM fraction: {r_smem[best_index]:.6g}")
    print(f"SMEM: {smem_bytes[best_index] / 2**20:.3f} MiB")
    print(f"CUDA Cores: {int(cuda_cores[best_index])}")
    print(f"Tensor Cores: {int(tensor_cores[best_index])}")
    print(f"Total execution time: {total_time[best_index] * 1e3:.6f} ms")
    print(f"Total HBM traffic: {total_hbm_traffic[best_index] / 2**20:.3f} MiB")
    print(f"Effective throughput: {effective_flops / 1e12:.3f} TFLOP/s")

    print("\n=== GEMM Stages ===")
    frontiers_by_aggregate: dict[str, list[TrafficFrontier]] = {}
    for frontier in frontiers:
        aggregate_name = aggregate_names[frontier.label]
        frontiers_by_aggregate.setdefault(aggregate_name, []).append(frontier)

    for name, time_seconds in task_times.items():
        print(f"\n{name}")
        print(f"  time: {time_seconds[best_index] * 1e3:.6f} ms")
        traffic = task_traffic[name][best_index]
        print(f"  HBM traffic: {traffic / 2**20:.3f} MiB")
        print(
            "  OI: "
            f"{task_operations_by_name[name] / traffic:.6f} FLOP/byte"
        )
        stage_frontiers = frontiers_by_aggregate.get(name, [])
        if len(stage_frontiers) == 1:
            print(
                "  mapping: "
                f"{format_selected_mapping(stage_frontiers[0], smem_bytes[best_index], tensor_roof[best_index])}"
            )
        elif stage_frontiers:
            print("  constituent mappings:")
            for stage_frontier in stage_frontiers:
                weight = group_weights[stage_frontier.label]
                print(
                    f"    {stage_frontier.label} "
                    f"(expected_count={weight:.8g}): "
                    f"{format_selected_mapping(stage_frontier, smem_bytes[best_index], tensor_roof[best_index])}"
                )

    print("\n=== Attention & Norm Stages ===")
    print(
        f"pre_attention_rmsnorm time: {pre_attention_rmsnorm_time[best_index] * 1e3:.6f} ms"
    )
    print(
        "pre_attention_rmsnorm HBM traffic: "
        f"{PRE_ATTENTION_RMSNORM_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    print(f"\nmla_attention time: {attention_time[best_index] * 1e3:.6f} ms")
    print(
        "mla_attention HBM traffic: "
        f"{ATTENTION_CORE_TASK.traffic_bytes / 2**20:.3f} MiB "
        f"({ATTENTION_CORE_TASK.traffic_bytes / 2**30:.3f} GiB)"
    )
    print(
        "  mapping: "
        f"{format_attention_mapping(ATTENTION_CORE_TASK, smem_bytes[best_index], tensor_roof[best_index], cuda_roof[best_index])}"
    )
    print(
        f"\npost_attention_residual_add time: {post_attention_residual_add_time[best_index] * 1e3:.6f} ms"
    )
    print(
        "post_attention_residual_add HBM traffic: "
        f"{POST_ATTENTION_RESIDUAL_ADD_TASK.traffic_bytes / 2**20:.3f} MiB"
    )

    print("\n=== FFN Vector / Reduction Stages ===")
    print(f"rmsnorm_square_reduction time: {rmsnorm_time[best_index] * 1e3:.6f} ms")
    print(
        "rmsnorm_square_reduction HBM traffic: "
        f"{RMSNORM_SQUARE_REDUCTION_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    print(f"activation time: {activation_time[best_index] * 1e3:.6f} ms")
    print(f"activation HBM traffic: {ACTIVATION_TASK.traffic_bytes / 2**20:.3f} MiB")
    print(
        "expert_weighted_sum time: "
        f"{expert_weighted_sum_time[best_index] * 1e3:.6f} ms"
    )
    print(
        "expert_weighted_sum HBM traffic: "
        f"{EXPERT_WEIGHTED_SUM_TASK.traffic_bytes / 2**20:.3f} MiB"
    )
    print(f"residual_add time: {residual_add_time[best_index] * 1e3:.6f} ms")
    print(f"residual_add HBM traffic: {RESIDUAL_ADD_TASK.traffic_bytes / 2**20:.3f} MiB")

    return results


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="GLM-5.2 prefill-layer die-area / latency estimator "
        "(pre-attn RMSNorm + causal MLA + residual + MoE FFN)."
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=DEFAULT_PREFILL_SEQUENCES,
        help=f"number of prompts prefilled together (default {DEFAULT_PREFILL_SEQUENCES}).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"prompt length in tokens (default {DEFAULT_SEQ_LEN}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="skip the (large) CSV and PNG outputs; print the report only.",
    )
    parser.add_argument(
        "--no-snowcat",
        action="store_true",
        help="disable the Snowcat traffic frontier; use algorithmic-minimum HBM "
        "traffic per GEMM and BW_eff = min(bw, SMEM/latency) (overly optimistic).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.no_snowcat:
        USE_SNOWCAT = False
    configure(args.sequences, args.seq_len)
    main(write_outputs=not args.no_write)
