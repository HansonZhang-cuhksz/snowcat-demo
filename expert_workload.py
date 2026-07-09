"""Even token-to-expert distribution with small-batch / tensor-core-tile handling.

Shared by all area-distribution analyzers (ffn_area.py, ffn_area_latency.py,
ffn_fused_area.py, ffn_fused_area_latency.py, decode_area_latency.py).

Model of even routing when batch*top_k is not a clean multiple of the expert count:

- ``total_assignments = batch_tokens * top_k`` token->expert slots.
- ``base, rem = divmod(total_assignments, experts)``.  ``rem`` experts get ``base+1``
  tokens (ceil), the remaining ``experts-rem`` get ``base`` tokens (floor).
- If ``base == 0`` (batch too small for every expert to receive a token), only ``rem``
  experts are active with 1 token each; the other ``experts-rem`` are idle (not computed).

Tensor cores need at least ``min_bm`` (=16) rows.  A per-expert GEMM whose real token
count is < 16 is padded to M=16 for the GEMM's tiling/traffic (tensor-core underutilized);
the activation/SwiGLU and expert-combine stages still use the real token counts.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MIN_TENSOR_BM = 16


@dataclass(frozen=True)
class EvenExpertSplit:
    batch_tokens: int
    experts: int
    top_k: int
    total_assignments: int          # batch_tokens * top_k
    active_experts: int             # experts receiving >= 1 token
    idle_experts: int               # experts receiving 0 tokens
    floor_tokens: int               # base = total // experts
    ceil_tokens: int                # base + 1 when rem else base
    num_ceil: int                   # experts receiving ceil_tokens (= rem)
    num_floor: int                  # active experts receiving floor_tokens (base >= 1)
    token_groups: tuple[tuple[int, int], ...]  # (tokens_per_expert > 0, expert_count)

    @property
    def uneven(self) -> bool:
        return self.total_assignments % self.experts != 0

    def summary(self) -> str:
        grp = ", ".join(
            f"{count} expert(s) x {tokens} token(s)" for tokens, count in self.token_groups
        )
        idle = f"; {self.idle_experts} idle" if self.idle_experts else ""
        return f"{self.active_experts}/{self.experts} experts active ({grp}){idle}"


def even_expert_token_split(
    batch_tokens: int, experts: int, top_k: int
) -> EvenExpertSplit:
    if batch_tokens <= 0 or experts <= 0 or top_k <= 0:
        raise ValueError("batch_tokens, experts and top_k must be positive")
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")

    total = batch_tokens * top_k
    base, rem = divmod(total, experts)

    token_groups: list[tuple[int, int]] = []
    num_floor = 0
    if base > 0:
        num_floor = experts - rem
        token_groups.append((base, experts - rem))
    if rem > 0:
        token_groups.append((base + 1, rem))

    active = sum(count for _, count in token_groups)
    ceil_tokens = base + 1 if rem else base
    return EvenExpertSplit(
        batch_tokens=batch_tokens,
        experts=experts,
        top_k=top_k,
        total_assignments=total,
        active_experts=active,
        idle_experts=experts - active,
        floor_tokens=base,
        ceil_tokens=ceil_tokens,
        num_ceil=rem,
        num_floor=num_floor,
        token_groups=tuple(token_groups),
    )


def padded_m(tokens: int, min_bm: int = DEFAULT_MIN_TENSOR_BM) -> int:
    """GEMM M for a per-expert GEMM: real token count, floored to the tensor-core
    minimum tile (16) so a valid BM>=16 tiling exists (tensor core underutilized)."""
    return tokens if tokens >= min_bm else min_bm


def padded_gemm_groups(
    split: EvenExpertSplit, min_bm: int = DEFAULT_MIN_TENSOR_BM
) -> list[tuple[int, int]]:
    """(M_padded, expert_count) for the per-expert GEMMs, merged by padded M.

    Token-count groups that pad to the same M (e.g. floor=8 and ceil=9 both -> 16)
    collapse into a single GEMM group.
    """
    merged: dict[int, int] = {}
    for tokens, count in split.token_groups:
        m = padded_m(tokens, min_bm)
        merged[m] = merged.get(m, 0) + count
    return sorted(merged.items())
