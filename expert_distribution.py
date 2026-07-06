from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertTokenDistribution:
    support: tuple[tuple[int, float], ...]
    model: str
    batch_tokens: int
    experts: int
    top_k: int
    selection_probability: float
    mean: float
    variance: float
    retained_probability_mass: float


def binomial_expert_token_distribution(
    batch_tokens: int,
    experts: int,
    top_k: int,
    probability_cutoff: float = 1e-12,
) -> ExpertTokenDistribution:
    """Return per-expert token-count probabilities for uniform random routing.

    Each token chooses ``top_k`` distinct experts uniformly.  For any one expert,
    a token is routed to that expert with probability p = top_k / experts.
    Therefore the token count for one expert is Binomial(batch_tokens, p).

    The returned support drops probabilities below ``probability_cutoff`` and
    renormalizes the retained probabilities.  The full-distribution mean and
    variance are still reported.
    """
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    if experts <= 0:
        raise ValueError("experts must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")
    if probability_cutoff < 0:
        raise ValueError("probability_cutoff must be non-negative")

    p = top_k / experts
    q = 1.0 - p
    mean = batch_tokens * p
    variance = batch_tokens * p * q

    if p == 1.0:
        return ExpertTokenDistribution(
            support=((batch_tokens, 1.0),),
            model="binomial",
            batch_tokens=batch_tokens,
            experts=experts,
            top_k=top_k,
            selection_probability=p,
            mean=mean,
            variance=variance,
            retained_probability_mass=1.0,
        )

    probability = q**batch_tokens
    retained: list[tuple[int, float]] = []
    retained_mass = 0.0
    for count in range(batch_tokens + 1):
        if probability >= probability_cutoff:
            retained.append((count, probability))
            retained_mass += probability
        if count == batch_tokens:
            break
        probability *= (batch_tokens - count) / (count + 1) * p / q

    if not retained:
        raise ValueError("probability_cutoff removed the entire distribution")

    normalized = tuple((count, prob / retained_mass) for count, prob in retained)
    return ExpertTokenDistribution(
        support=normalized,
        model="binomial",
        batch_tokens=batch_tokens,
        experts=experts,
        top_k=top_k,
        selection_probability=p,
        mean=mean,
        variance=variance,
        retained_probability_mass=retained_mass,
    )
