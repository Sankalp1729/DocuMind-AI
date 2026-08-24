from __future__ import annotations

from typing import Dict, List, Set

from backend.evaluation.metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class Evaluator:
    """Evaluate ranked retrieval results against relevance judgments."""

    def __init__(self, k: int = 10):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def evaluate(
        self,
        queries: List[str],
        retrieved_lists: List[List[int]],
        ground_truths: List[Set[int]],
    ) -> Dict[str, float]:
        if len(retrieved_lists) != len(ground_truths):
            raise ValueError("retrieved_lists and ground_truths must have equal length")
        if queries and len(queries) != len(retrieved_lists):
            raise ValueError("queries and retrieval results must have equal length")
        if not retrieved_lists:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "map": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
            }

        precisions = [
            precision_at_k(retrieved, relevant, self.k)
            for retrieved, relevant in zip(retrieved_lists, ground_truths)
        ]
        recalls = [
            recall_at_k(retrieved, relevant, self.k)
            for retrieved, relevant in zip(retrieved_lists, ground_truths)
        ]
        average_precisions = [
            average_precision(retrieved, relevant, self.k)
            for retrieved, relevant in zip(retrieved_lists, ground_truths)
        ]
        ndcgs = [
            ndcg_at_k(retrieved, relevant, self.k)
            for retrieved, relevant in zip(retrieved_lists, ground_truths)
        ]

        return {
            "precision_at_k": sum(precisions) / len(precisions),
            "recall_at_k": sum(recalls) / len(recalls),
            "map": sum(average_precisions) / len(average_precisions),
            "mrr": mean_reciprocal_rank(retrieved_lists, ground_truths),
            "ndcg": sum(ndcgs) / len(ndcgs),
        }
