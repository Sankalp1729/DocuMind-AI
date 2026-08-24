from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

from backend.retrieval.retrieval_fusion import reciprocal_rank_fusion


RankedItem = Tuple[str, float]


class RetrievalStrategy(str, Enum):
    """Supported retrieval configurations for controlled RAG experiments."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"


@dataclass(frozen=True)
class StrategyResult:
    strategy: RetrievalStrategy
    ranked_ids: List[str]


def select_strategy(
    strategy: RetrievalStrategy | str,
    *,
    dense_results: Sequence[RankedItem],
    bm25_results: Sequence[RankedItem],
    top_k: int,
) -> StrategyResult:
    """Select a deterministic retrieval ranking for an experiment.

    Reranking is intentionally handled by the caller because a cross-encoder
    needs the query and candidate document text. This function therefore makes
    the four experiment modes explicit without hiding model inference inside
    ranking logic.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    strategy = RetrievalStrategy(strategy)
    if strategy == RetrievalStrategy.DENSE:
        ranked = dense_results
    elif strategy == RetrievalStrategy.BM25:
        ranked = bm25_results
    else:
        ranked = reciprocal_rank_fusion([
            list(dense_results),
            list(bm25_results),
        ])

    return StrategyResult(
        strategy=strategy,
        ranked_ids=[str(doc_id) for doc_id, _score in list(ranked)[:top_k]],
    )


def strategy_matrix() -> List[Dict[str, str]]:
    """Return the canonical experiment matrix used by benchmark tooling."""
    return [
        {"strategy": RetrievalStrategy.DENSE.value, "reranker": "disabled"},
        {"strategy": RetrievalStrategy.BM25.value, "reranker": "disabled"},
        {"strategy": RetrievalStrategy.HYBRID.value, "reranker": "disabled"},
        {"strategy": RetrievalStrategy.HYBRID_RERANKED.value, "reranker": "cross-encoder"},
    ]
