from pathlib import Path

from scripts.run_benchmark_matrix import render_summary, run_matrix


class FakeEvaluationService:
    def load_dataset(self, dataset_name):
        return object()

    def run_dataset_benchmark(self, rag_service, dataset_name, benchmark_id, top_k):
        from types import SimpleNamespace

        metrics = {
            f"precision_at_{top_k}": 0.8,
            f"recall_at_{top_k}": 0.9,
            "map": 0.85,
            "mrr": 0.9,
            "ndcg": 0.88,
        }
        return SimpleNamespace(
            top_k=top_k,
            retrieval_metrics=metrics,
            groundedness_rate=1.0,
            retrieval_latency_ms=12.5,
            total_latency_ms=20.0,
            model_dump=lambda mode="json": {
                "top_k": top_k,
                "retrieval_metrics": metrics,
                "groundedness_rate": 1.0,
                "retrieval_latency_ms": 12.5,
                "total_latency_ms": 20.0,
            },
        )


def test_matrix_deduplicates_cutoffs_and_persists(monkeypatch, tmp_path: Path):
    fake_service = FakeEvaluationService()
    monkeypatch.setattr("scripts.run_benchmark_matrix.EvaluationService", lambda storage_dir: fake_service)
    monkeypatch.setattr("scripts.run_benchmark_matrix.build_rag_service", lambda: object())

    results = run_matrix("demo", [3, 5, 5, 10], tmp_path)

    assert [result["top_k"] for result in results] == [3, 5, 10]
    assert (tmp_path / "demo-k3.json").exists()
    assert (tmp_path / "demo-k5.json").exists()
    assert (tmp_path / "demo-k10.json").exists()


def test_summary_uses_actual_cutoff_keys():
    results = [
        {
            "top_k": 5,
            "retrieval_metrics": {
                "precision_at_5": 0.8,
                "recall_at_5": 0.9,
                "map": 0.85,
                "mrr": 0.9,
                "ndcg": 0.88,
            },
            "groundedness_rate": 1.0,
            "retrieval_latency_ms": 12.5,
            "total_latency_ms": 20.0,
        }
    ]

    summary = render_summary(results)

    assert "| 5 | 0.8000 | 0.9000 |" in summary
    assert "0.8500" in summary
