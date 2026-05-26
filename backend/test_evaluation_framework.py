from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backend.services.evaluation_service import EvaluationService


class FakeDoc:
    def __init__(self, page_content: str, source_file: str, page: int = 1):
        self.page_content = page_content
        self.metadata = {"source_file": source_file, "page": page}


class FakeRetrievalExplanation:
    def __init__(self):
        self.latency_ms = 12.5
        self.stage_timings_ms = {"reranking_ms": 4.2, "vector_retrieval_ms": 3.1}

    def model_dump(self):
        return {
            "latency_ms": self.latency_ms,
            "stage_timings_ms": self.stage_timings_ms,
        }


class FakeRagService:
    def retrieve(self, question: str):
        return (
            "The report explains the retrieval pipeline and evaluation history.",
            [FakeDoc("The report explains the retrieval pipeline and evaluation history.", "report.pdf", 1)],
            FakeRetrievalExplanation(),
        )


def test_dataset_benchmark_runs_and_persists(tmp_path: Path) -> None:
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
    assert result.retrieval_metrics["precision_at_10"] == 1.0
    assert result.retrieval_metrics["recall_at_10"] == 1.0
    assert result.retrieval_metrics["mrr"] == 1.0
    assert result.retrieval_metrics["ndcg"] == 1.0
    assert result.groundedness_rate == 1.0
    assert Path(saved_path).exists()

    history = service.load_benchmark_history()
    leaderboard = service.leaderboard()
    dashboard = service.benchmark_dashboard()

    assert history
    assert leaderboard
    assert dashboard["runs"] == 1
    assert dashboard["leaderboard"]
