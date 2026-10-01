"""Reciprocal Rank Fusion (RRF).

Pure function combining multiple ranked lists into a single ranking.
Original formula (Cormack et al., 2009):

    RRF.score(d) = sum_over_sources( 1 / (k + rank_in_source(d)) )

where k is a constant (default 60) that dampens the impact of high ranks.

Adapted from mlaw-rag backend/src/domain/rrf_fusion.py.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RankedItem:
    """A single item with its rank in one source.

    `id` is the unique identifier (chunk_id, item_id, etc.).
    `score` is the original score from the source (optional, for diagnostics).
    `rank` is 1-indexed rank within this source.
    """
    id: str
    rank: int
    score: float = 0.0


@dataclass(slots=True, frozen=True)
class FusedItem:
    """Output item after RRF fusion."""
    id: str
    fused_rank: int
    rrf_score: float


def rrf_fusion(
    sources: Sequence[Sequence[RankedItem]],
    k: int = 60,
) -> list[FusedItem]:
    """Combine multiple ranked lists into one fused ranking.

    Args:
        sources: Each source is a sequence of RankedItem, sorted by relevance (rank 1 = best).
        k: Smoothing constant. Default 60 per literature. Must be >= 1.

    Returns:
        List of FusedItem, sorted by RRF score descending (best first).

    Examples:
        >>> sources = [
        ...     [RankedItem("a", 1), RankedItem("b", 2)],
        ...     [RankedItem("b", 1), RankedItem("a", 2)],
        ... ]
        >>> result = rrf_fusion(sources)
        >>> result[0].id in ("a", "b")  # tied
        True
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Accumulate RRF scores per item
    scores: dict[str, float] = {}
    for src in sources:
        for item in src:
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + item.rank)

    # Sort descending by score; break ties alphabetically for determinism
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedItem(id=i, fused_rank=r + 1, rrf_score=round(s, 6))
        for r, (i, s) in enumerate(ordered)
    ]
