from __future__ import annotations

import math
from typing import List


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def precision_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    _validate_k(k)
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for doc in retrieved_k if doc in relevant) / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    _validate_k(k)
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return min(1.0, sum(1 for doc in retrieved_k if doc in relevant) / len(relevant))


def average_precision(retrieved: List[int], relevant: set, k: int | None = None) -> float:
    """Average precision over the ranked list, optionally truncated at k.

    The denominator is the number of relevant documents, so missed relevant
    documents reduce AP instead of disappearing from the score.
    """
    if k is not None:
        _validate_k(k)
        retrieved = retrieved[:k]
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            score += hits / rank

    return score / len(relevant) if hits else 0.0


def mean_reciprocal_rank(all_retrieved: List[List[int]], all_relevant: List[set]) -> float:
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("retrieved and relevant collections must have equal length")

    reciprocal_ranks = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        reciprocal_rank = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def dcg(retrieved: List[int], relevant: set, k: int) -> float:
    _validate_k(k)
    score = 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        relevance = 1.0 if doc in relevant else 0.0
        score += (2**relevance - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    _validate_k(k)
    if not relevant:
        return 0.0
    idcg = dcg(list(relevant), relevant, k)
    if idcg == 0:
        return 0.0
    return min(1.0, dcg(retrieved, relevant, k) / idcg)
