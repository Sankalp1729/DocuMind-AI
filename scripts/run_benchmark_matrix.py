from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from backend.core.config import DATA_DIR
from backend.services.evaluation_service import EvaluationService
from backend.services.rag_service import RagService
from backend.services.retrieval_service import RetrievalService
from backend.services.telemetry_service import TelemetryService
from backend.services.vector_store_service import VectorStoreService


def build_rag_service() -> RagService:
    """Build the production retrieval stack without starting the API server."""
    vector_store_service = VectorStoreService()
    telemetry_service = TelemetryService()
    retrieval_service = RetrievalService(
        vector_store_service=vector_store_service,
        telemetry_service=telemetry_service,
    )
    return RagService(
        vector_store_service=vector_store_service,
        retrieval_service=retrieval_service,
        telemetry_service=telemetry_service,
        cache_service=None,
    )


def run_matrix(
    dataset_name: str,
    top_k_values: Iterable[int],
    output_dir: Path,
) -> List[dict]:
    """Run the same benchmark dataset at several retrieval cutoffs."""
    ks = list(dict.fromkeys(int(k) for k in top_k_values))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("top_k_values must contain positive integers")

    evaluation_service = EvaluationService(storage_dir=DATA_DIR)
    if evaluation_service.load_dataset(dataset_name) is None:
        raise FileNotFoundError(f"Benchmark dataset not found: {dataset_name}")

    rag_service = build_rag_service()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    for top_k in ks:
        result = evaluation_service.run_dataset_benchmark(
            rag_service,
            dataset_name,
            benchmark_id=f"{dataset_name}-k{top_k}",
            top_k=top_k,
        )
        payload = result.model_dump(mode="json")
        output_path = output_dir / f"{dataset_name}-k{top_k}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        results.append(payload)

    return results


def render_summary(results: List[dict]) -> str:
    lines = [
        "| K | Precision | Recall | MAP | MRR | nDCG | Groundedness | Retrieval ms | Total ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        k = int(result["top_k"])
        metrics = result["retrieval_metrics"]
        lines.append(
            f"| {k} | {metrics.get(f'precision_at_{k}', 0.0):.4f} | "
            f"{metrics.get(f'recall_at_{k}', 0.0):.4f} | "
            f"{metrics.get('map', 0.0):.4f} | {metrics.get('mrr', 0.0):.4f} | "
            f"{metrics.get('ndcg', 0.0):.4f} | {result['groundedness_rate']:.4f} | "
            f"{result['retrieval_latency_ms']:.2f} | {result['total_latency_ms']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DocuMind RAG benchmark across multiple K values.")
    parser.add_argument("--dataset", default="example_rag_benchmark")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "benchmark_matrix")
    args = parser.parse_args()

    try:
        results = run_matrix(args.dataset, args.top_k, args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(f"\nRAG benchmark matrix: {args.dataset}")
    print(render_summary(results))
    print(f"\nPersisted {len(results)} benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
