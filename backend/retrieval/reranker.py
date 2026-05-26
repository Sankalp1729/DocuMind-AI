from __future__ import annotations

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranker using a cross-encoder model from sentence-transformers.

    Usage: instantiate with a lightweight cross-encoder, call `rerank(query, candidates)`.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_texts: List[str]) -> List[Tuple[int, float]]:
        # returns list of (index, score) sorted descending
        if not candidate_texts:
            return []

        pairs = [[query, c] for c in candidate_texts]
        scores = self.model.predict(pairs)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed
