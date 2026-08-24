from pathlib import Path
import json

from backend.services.evaluation_service import EvaluationService


class FakeRagService:
    def retrieve(self, question: str, top_k: int = 10):
        from types import SimpleNamespace

        del question
        docs = [
            SimpleNamespace(metadata={"source_file": "report", "page": 1}, page_content="retrieval pipeline")
        ] * top_k
        explanation = SimpleNamespace(latency_ms=1.0, model_dump=lambda: {"stage_timings_ms": {"reranking_ms": 0.0}})
        return "retrieval pipeline", docs, explanation


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


def test_dataset_benchmark_runs_and_persists(tmp_path: Path, monkeypatch) -> None:
    _grounded(monkeypatch)
    service = EvaluationService(storage_dir=tmp_path)
    dataset_dir = tmp_path / "evaluation_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_payload = {
        "dataset_name": "unit_test_dataset",
        "description": "Synthetic dataset for testing the evaluation framework.",
        "version": "1.0",
        "queries": [
            {
                "query_id": "q1",
                "question": "What does the report explain?",
                "relevant_sources": ["report"],
                "relevant_pages": [1],
                "answer_references": ["retrieval pipeline"],
            }
        ],
    }
    (dataset_dir / "unit_test_dataset.json").write_text(json.dumps(dataset_payload), encoding="utf-8")

    result = service.run_dataset_benchmark(
        FakeRagService(),
        "unit_test_dataset",
        benchmark_id="bench-1",
        top_k=1,
        answer_fn=lambda context, question: "The report explains the retrieval pipeline and evaluation history.",
    )
    saved_path = service.save_benchmark_result("bench-1", result)

    assert result.dataset_name == "unit_test_dataset"
    assert result.num_queries == 1
    assert result.top_k == 1
    assert result.retrieval_metrics["precision_at_1"] == 1.0
    assert result.retrieval_metrics["recall_at_1"] == 1.0
    assert "precision_at_10" not in result.retrieval_metrics
    assert "recall_at_10" not in result.retrieval_metrics
    assert result.retrieval_metrics["mrr"] == 1.0
    assert result.retrieval_metrics["ndcg"] == 1.0
    assert result.groundedness_rate == 1.0
    assert Path(saved_path).exists()

    history = service.load_benchmark_history()
    leaderboard = service.leaderboard()
    dashboard = service.benchmark_dashboard()
    assert len(history) == 1
    assert len(leaderboard) == 1
    assert dashboard["count"] == 1
