from __future__ import annotations

from typing import List, Tuple, Dict


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each input list is a list of (doc_id, score) ordered by rank (best first).
    Returns a list of (doc_id, rrf_score) ordered descending.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged
