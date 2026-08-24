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


def test_non_ten_cutoff_is_not_written_as_precision_at_10(monkeypatch) -> None:
    dataset = BenchmarkDataset.model_validate(
        {
            "dataset_name": "cutoff_contract",
            "description": "Tests metric key semantics.",
            "queries": [
                {
                    "query_id": "q1",
                    "question": "What is in the report?",
                    "relevant_sources": ["report.pdf"],
                }
            ],
        }
    )

    monkeypatch.setattr(
        "backend.evaluation.benchmark_runner.GroundednessScorer.score_groundedness",
        lambda self, answer, passages: {
            "is_grounded": True,
            "confidence": 1.0,
            "hallucination_risk": "low",
            "unsupported_claims": [],
        },
    )

    result = run_benchmark(
        FakeRagService(),
        dataset,
        top_k=5,
        answer_fn=lambda context, question: "The report contains the answer.",
    )

    assert result.top_k == 5
    assert "precision_at_5" in result.retrieval_metrics
    assert "recall_at_5" in result.retrieval_metrics
    assert "precision_at_10" not in result.retrieval_metrics
    assert "recall_at_10" not in result.retrieval_metrics


def test_ten_cutoff_keeps_legacy_dashboard_metric_keys(monkeypatch) -> None:
    dataset = BenchmarkDataset.model_validate(
        {
            "dataset_name": "cutoff_contract",
            "description": "Tests legacy metric compatibility.",
            "queries": [
                {
                    "query_id": "q1",
                    "question": "What is in the report?",
                    "relevant_sources": ["report.pdf"],
                }
            ],
        }
    )

    monkeypatch.setattr(
        "backend.evaluation.benchmark_runner.GroundednessScorer.score_groundedness",
        lambda self, answer, passages: {
            "is_grounded": True,
            "confidence": 1.0,
            "hallucination_risk": "low",
            "unsupported_claims": [],
        },
    )

    result = run_benchmark(
        FakeRagService(),
        dataset,
        top_k=10,
        answer_fn=lambda context, question: "The report contains the answer.",
    )

    assert result.retrieval_metrics["precision_at_10"] == result.retrieval_metrics["precision_at_10"]
    assert result.retrieval_metrics["recall_at_10"] == result.retrieval_metrics["recall_at_10"]
