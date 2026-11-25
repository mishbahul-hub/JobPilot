"""Ranking metrics with binary relevance."""

import math


def precision_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for rank, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0
