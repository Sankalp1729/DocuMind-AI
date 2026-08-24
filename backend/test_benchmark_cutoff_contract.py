from types import SimpleNamespace

from backend.evaluation.benchmark_runner import run_benchmark
from backend.schemas.evaluation import BenchmarkDataset


class FakeRagService:
    def retrieve(self, question: str, top_k: int = 10):
        del question
        docs = [
            SimpleNamespace(metadata={"source_file": "report.pdf", "page": 1}, page_content="relevant passage")
        ] * top_k
        explanation = SimpleNamespace(latency_ms=1.0, model_dump=lambda: {"stage_timings_ms": {"reranking_ms": 0.0}})
        return "context", docs, explanation


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset.model_validate(
        {
            "dataset_name": "cutoff_contract",
            "description": "Tests benchmark metric key semantics.",
            "queries": [
                {
                    "query_id": "q1",
                    "question": "What is in the report?",
                    "relevant_sources": ["report.pdf"],
                }
            ],
        }
    )


def _grounded(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.evaluation.benchmark_runner.GroundednessScorer.score_groundedness",
        lambda self, answer, passages: {
            "is_grounded": True,
            "confidence": 1.0,
            "hallucination_risk": "low",
            "unsupported_claims": [],
            "reasoning": "The answer is directly supported by the retrieved passage.",
        },
    )


def test_non_ten_cutoff_is_not_written_as_precision_at_10(monkeypatch) -> None:
    _grounded(monkeypatch)
    result = run_benchmark(
        FakeRagService(), _dataset(), top_k=5,
        answer_fn=lambda context, question: "The report contains the answer.",
    )

    assert result.top_k == 5
    assert result.retrieval_metrics["precision_at_5"] == 1.0
    assert result.retrieval_metrics["recall_at_5"] == 1.0
    assert "precision_at_10" not in result.retrieval_metrics
    assert "recall_at_10" not in result.retrieval_metrics


def test_ten_cutoff_keeps_legacy_dashboard_metric_keys(monkeypatch) -> None:
    _grounded(monkeypatch)
    result = run_benchmark(
        FakeRagService(), _dataset(), top_k=10,
        answer_fn=lambda context, question: "The report contains the answer.",
    )

    assert result.retrieval_metrics["precision_at_10"] == 1.0
    assert result.retrieval_metrics["recall_at_10"] == 1.0
