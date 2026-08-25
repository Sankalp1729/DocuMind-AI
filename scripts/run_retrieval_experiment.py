from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.core.config import DATA_DIR
from backend.retrieval.controlled_retrieval import ControlledRetrieval
from backend.retrieval.strategy import RetrievalStrategy
from backend.services.evaluation_service import EvaluationService
from backend.services.retrieval_service import RetrievalService
from backend.services.telemetry_service import TelemetryService
from backend.services.vector_store_service import VectorStoreService


def build_retrieval_service() -> RetrievalService:
    vector_store_service = VectorStoreService()
    return RetrievalService(
        vector_store_service=vector_store_service,
        telemetry_service=TelemetryService(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one controlled DocuMind retrieval experiment.")
    parser.add_argument("--dataset", default="example_rag_benchmark")
    parser.add_argument("--strategy", choices=[strategy.value for strategy in RetrievalStrategy], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    evaluation = EvaluationService(storage_dir=DATA_DIR)
    dataset = evaluation.load_dataset(args.dataset)
    if dataset is None:
        parser.error(f"Benchmark dataset not found: {args.dataset}")

    service = build_retrieval_service()
    runner = ControlledRetrieval(service)
    strategy = RetrievalStrategy(args.strategy)
    rows = []

    for query in dataset.queries:
        docs, info = runner.retrieve(query.question, args.top_k, strategy)
        rows.append({
            "query_id": query.query_id,
            "strategy": strategy.value,
            "top_k": args.top_k,
            "retrieved": len(docs),
            "source_files": [doc.metadata.get("source_file") for doc in docs],
            "pages": [doc.metadata.get("page") for doc in docs],
            "pipeline": info,
        })

    result = {
        "dataset_name": dataset.dataset_name,
        "strategy": strategy.value,
        "top_k": args.top_k,
        "num_queries": len(rows),
        "queries": rows,
    }

    output = args.output or (DATA_DIR / "evaluation" / f"retrieval_{args.dataset}_{strategy.value}_k{args.top_k}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Retrieval experiment written to {output}")


if __name__ == "__main__":
    main()
