from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _metric(metrics: Dict[str, Any], name: str, top_k: int) -> float:
    return float(metrics.get(f"{name}_at_{top_k}", metrics.get(f"{name}_at_10", 0.0)))


def summarize_benchmark(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable, presentation-friendly summary of one benchmark run."""
    top_k = int(result.get("top_k", 10))
    metrics = result.get("retrieval_metrics") or result.get("metrics") or {}
    return {
        "benchmark_id": result.get("benchmark_id") or result.get("run_id"),
        "dataset_name": result.get("dataset_name") or result.get("dataset"),
        "timestamp": result.get("timestamp"),
        "top_k": top_k,
        "precision_at_k": _metric(metrics, "precision", top_k),
        "recall_at_k": _metric(metrics, "recall", top_k),
        "map": float(metrics.get("map", 0.0)),
        "mrr": float(metrics.get("mrr", 0.0)),
        "ndcg": float(metrics.get("ndcg", 0.0)),
        "groundedness_rate": float(result.get("groundedness_rate", 0.0)),
        "hallucination_rate": float(result.get("hallucination_rate", 0.0)),
        "retrieval_latency_ms": float(result.get("retrieval_latency_ms", 0.0)),
        "total_latency_ms": float(result.get("total_latency_ms", 0.0)),
    }


def load_results(directory: Path) -> List[Dict[str, Any]]:
    """Load persisted benchmark JSON files in deterministic order."""
    results: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("benchmark_*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def render_markdown(results: Iterable[Dict[str, Any]]) -> str:
    summaries = [summarize_benchmark(result) for result in results]
    lines = [
        "# RAG Benchmark Report",
        "",
        "| Benchmark | Dataset | K | Precision | Recall | MAP | MRR | nDCG | Grounded | Hallucination | Retrieval ms | Total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {benchmark_id} | {dataset_name} | {top_k} | {precision_at_k:.3f} | {recall_at_k:.3f} | "
            "{map:.3f} | {mrr:.3f} | {ndcg:.3f} | {groundedness_rate:.3f} | "
            "{hallucination_rate:.3f} | {retrieval_latency_ms:.1f} | {total_latency_ms:.1f} |".format(**item)
        )
    lines.append("")
    lines.append("Metrics are generated from persisted benchmark results; no performance values are hard-coded.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render persisted DocuMind RAG benchmark results as Markdown.")
    parser.add_argument("--directory", type=Path, default=Path("data/evaluations"), help="Directory containing benchmark_*.json files")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path")
    args = parser.parse_args()

    report = render_markdown(load_results(args.directory))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
