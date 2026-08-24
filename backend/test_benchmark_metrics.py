import pytest

from backend.evaluation.benchmark_runner import (
    _average_precision,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
)
from backend.evaluation.evaluator import Evaluator
from backend.evaluation.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k


def test_average_precision_rewards_early_relevant_results() -> None:
    assert _average_precision([1, 0, 1], relevant_count=2) == pytest.approx(5 / 6)
    assert _average_precision([1, 1], relevant_count=2) == pytest.approx(1.0)
    assert _average_precision([1, 0, 1], relevant_count=2) > _average_precision([0, 1, 1], relevant_count=2)


def test_average_precision_penalizes_missed_relevant_documents() -> None:
    assert average_precision([1, 0, 0], {1, 2}) == pytest.approx(0.5)
    assert average_precision([1, 0, 2], {1, 2}) == pytest.approx(5 / 6)
    assert average_precision([1, 0, 2], {1, 2, 3}) == pytest.approx(5 / 9)


def test_precision_and_recall_are_bounded() -> None:
    assert _precision_at_k([1, 1, 0], 2) == 1.0
    assert _recall_at_k([1, 1, 1], relevant_count=2, k=3) == 1.0
    assert _recall_at_k([0, 1, 0], relevant_count=2, k=3) == 0.5
    assert recall_at_k([1, 1, 1], {1, 2}, 3) == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert _mrr([0, 0, 1]) == 1 / 3
    assert _mrr([0, 1, 1]) == 0.5


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert _ndcg_at_k([1, 1, 0], 3) == 1.0
    assert ndcg_at_k([1, 2, 9], {1, 2}, 2) == 1.0


def test_metric_functions_reject_invalid_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k([1], {1}, 0)
    with pytest.raises(ValueError):
        Evaluator(k=0)


def test_evaluator_validates_collection_lengths() -> None:
    evaluator = Evaluator(k=5)
    with pytest.raises(ValueError):
        evaluator.evaluate(["q1"], [[1], [2]], [{1}, {2}])
    with pytest.raises(ValueError):
        evaluator.evaluate(["q1"], [[1]], [{1}, {2}])


def test_evaluator_handles_empty_input() -> None:
    assert Evaluator().evaluate([], [], []) == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "map": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
