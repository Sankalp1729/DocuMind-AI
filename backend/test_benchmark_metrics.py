from backend.evaluation.benchmark_runner import (
    _average_precision,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
)


def test_average_precision_rewards_early_relevant_results() -> None:
    assert _average_precision([1, 0, 1], relevant_count=2) == 1.0
    assert _average_precision([0, 1, 1], relevant_count=2) == 2 / 3


def test_precision_and_recall_are_bounded() -> None:
    assert _precision_at_k([1, 1, 0], 2) == 1.0
    assert _recall_at_k([1, 1, 1], relevant_count=2, k=3) == 1.0
    assert _recall_at_k([0, 1, 0], relevant_count=2, k=3) == 0.5


def test_mrr_uses_first_relevant_rank() -> None:
    assert _mrr([0, 0, 1]) == 1 / 3
    assert _mrr([0, 1, 1]) == 0.5


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert _ndcg_at_k([1, 1, 0], 3) == 1.0
