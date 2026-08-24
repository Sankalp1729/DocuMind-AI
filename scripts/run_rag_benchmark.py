from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.core.config import DATA_DIR
from backend.services.evaluation_service import EvaluationService
from backend.services.rag_service import RagService
from backend.services.retrieval_service import RetrievalService
from backend.services.telemetry_service import TelemetryService
from backend.services.vector_store_service import VectorStoreService


def build_rag_service() -> RagService:
    """Build the same retrieval stack used by the API without starting FastAPI."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible DocuMind RAG benchmark.")
    parser.add_argument("--dataset", default="example_rag_benchmark", help="Benchmark dataset name")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval cutoff to evaluate")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the persisted benchmark JSON result",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    evaluation_service = EvaluationService(storage_dir=DATA_DIR)
    dataset = evaluation_service.load_dataset(args.dataset)
    if dataset is None:
        parser.error(f"Benchmark dataset not found: {args.dataset}")

    rag_service = build_rag_service()
    result = evaluation_service.run_dataset_benchmark(
        rag_service,
        args.dataset,
        benchmark_id=f"{args.dataset}-k{args.top_k}",
        top_k=args.top_k,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        print(f"Benchmark result written to {args.output}")
    else:
        output_path = evaluation_service.save_benchmark_dashboard_result(result)
        print(f"Benchmark result persisted to {output_path}")

    metrics = result.retrieval_metrics
    print("\nRAG Benchmark")
    print(f"dataset: {result.dataset_name}")
    print(f"top_k: {result.top_k}")
    print(f"queries: {result.num_queries}")
    print(f"precision@{result.top_k}: {metrics.get(f'precision_at_{result.top_k}', 0.0):.4f}")
    print(f"recall@{result.top_k}: {metrics.get(f'recall_at_{result.top_k}', 0.0):.4f}")
    print(f"MAP: {metrics.get('map', 0.0):.4f}")
    print(f"MRR: {metrics.get('mrr', 0.0):.4f}")
    print(f"nDCG@{result.top_k}: {metrics.get('ndcg', 0.0):.4f}")
    print(f"groundedness: {result.groundedness_rate:.4f}")
    print(f"hallucination risk: {result.hallucination_rate:.4f}")
    print(f"retrieval latency ms: {result.retrieval_latency_ms:.2f}")
    print(f"total latency ms: {result.total_latency_ms:.2f}")


if __name__ == "__main__":
    main()
