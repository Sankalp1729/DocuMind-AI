from __future__ import annotations

from typing import List, Dict
import math


def precision_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for r in retrieved_k if r in relevant) / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved_k if r in relevant) / len(relevant)


def average_precision(retrieved: List[int], relevant: set) -> float:
    hit_count = 0
    score = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hit_count += 1
            score += hit_count / i
    if hit_count == 0:
        return 0.0
    return score / hit_count


def mean_reciprocal_rank(all_retrieved: List[List[int]], all_relevant: List[set]) -> float:
    rr = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        rank = 0
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rank = 1.0 / i
                break
        rr.append(rank)
    return sum(rr) / len(rr) if rr else 0.0


def dcg(retrieved: List[int], relevant: set, k: int) -> float:
    dcg_score = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        rel = 1.0 if doc in relevant else 0.0
        dcg_score += (2 ** rel - 1) / math.log2(i + 1)
    return dcg_score


def ndcg_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    ideal = sorted(list(relevant))[:k]
    idcg = dcg(ideal, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg(retrieved, relevant, k) / idcg
