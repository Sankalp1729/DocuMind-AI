from pathlib import Path

from backend.evaluation.report import load_results, render_markdown, summarize_benchmark


def _result(top_k: int = 5):
    return {
        "benchmark_id": "bench-1",
        "dataset_name": "example",
        "timestamp": "2026-08-24T10:00:00+00:00",
        "top_k": top_k,
        "retrieval_metrics": {
            f"precision_at_{top_k}": 0.8,
            f"recall_at_{top_k}": 0.6,
            "map": 0.7,
            "mrr": 0.75,
            "ndcg": 0.72,
        },
        "groundedness_rate": 0.9,
        "hallucination_rate": 0.1,
        "retrieval_latency_ms": 12.5,
        "total_latency_ms": 120.0,
    }


def test_summarize_benchmark_uses_recorded_top_k() -> None:
    summary = summarize_benchmark(_result(top_k=5))

    assert summary["top_k"] == 5
    assert summary["precision_at_k"] == 0.8
    assert summary["recall_at_k"] == 0.6
    assert summary["mrr"] == 0.75


def test_render_markdown_contains_measured_metrics() -> None:
    report = render_markdown([_result()])

    assert "# RAG Benchmark Report" in report
    assert "bench-1" in report
    assert "0.800" in report
    assert "0.600" in report
    assert "Metrics are generated from persisted benchmark results" in report


def test_load_results_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "benchmark_b.json").write_text('{"benchmark_id":"b"}', encoding="utf-8")
    (tmp_path / "benchmark_a.json").write_text('{"benchmark_id":"a"}', encoding="utf-8")

    results = load_results(tmp_path)

    assert [result["benchmark_id"] for result in results] == ["a", "b"]
