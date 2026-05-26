from __future__ import annotations

from typing import List, Set, Dict
from backend.evaluation.metrics import precision_at_k, recall_at_k, average_precision, mean_reciprocal_rank, ndcg_at_k


class Evaluator:
    def __init__(self, k: int = 10):
        self.k = k

    def evaluate(self, queries: List[str], retrieved_lists: List[List[int]], ground_truths: List[Set[int]]) -> Dict:
        precisions = []
        recalls = []
        aps = []
        retrieved_ids = []

        for retrieved, relevant in zip(retrieved_lists, ground_truths):
            precisions.append(precision_at_k(retrieved, relevant, self.k))
            recalls.append(recall_at_k(retrieved, relevant, self.k))
            aps.append(average_precision(retrieved, relevant))
            retrieved_ids.append(retrieved)

        mrr = mean_reciprocal_rank(retrieved_lists, ground_truths)
        ndcg = sum(ndcg_at_k(r, g, self.k) for r, g in zip(retrieved_lists, ground_truths)) / len(retrieved_lists)

        return {
            "precision_at_k": sum(precisions) / len(precisions) if precisions else 0,
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0,
            "map": sum(aps) / len(aps) if aps else 0,
            "mrr": mrr,
            "ndcg": ndcg,
        }
