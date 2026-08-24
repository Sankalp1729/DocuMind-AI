import pytest

from backend.retrieval.strategy import RetrievalStrategy, select_strategy, strategy_matrix


DENSE = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
BM25 = [("b", 12.0), ("d", 10.0), ("a", 8.0)]


def test_dense_strategy_preserves_dense_order() -> None:
    result = select_strategy(RetrievalStrategy.DENSE, dense_results=DENSE, bm25_results=BM25, top_k=2)
    assert result.ranked_ids == ["a", "b"]


def test_bm25_strategy_preserves_bm25_order() -> None:
    result = select_strategy("bm25", dense_results=DENSE, bm25_results=BM25, top_k=2)
    assert result.ranked_ids == ["b", "d"]


def test_hybrid_strategy_uses_rrf() -> None:
    result = select_strategy("hybrid", dense_results=DENSE, bm25_results=BM25, top_k=4)
    assert result.ranked_ids[0] == "a" or result.ranked_ids[0] == "b"
    assert set(result.ranked_ids) == {"a", "b", "c", "d"}


def test_strategy_matrix_contains_controlled_experiments() -> None:
    matrix = strategy_matrix()
    assert [row["strategy"] for row in matrix] == [
        "dense",
        "bm25",
        "hybrid",
        "hybrid_reranked",
    ]
    assert matrix[-1]["reranker"] == "cross-encoder"


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        select_strategy("dense", dense_results=DENSE, bm25_results=BM25, top_k=0)
