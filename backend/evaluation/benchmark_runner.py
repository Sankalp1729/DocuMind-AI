from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.evaluation.groundedness import GroundednessScorer
from backend.schemas.evaluation import BenchmarkDataset, BenchmarkQueryResult, BenchmarkResult
from backend.schemas.retrieval import GroundednessScore


@dataclass(slots=True)
class BenchmarkRunContext:
    benchmark_id: str
    dataset_name: str
    timestamp: datetime
    query_results: List[BenchmarkQueryResult]
    metrics: Dict[str, float]
    retrieval_latency_ms: float
    reranking_latency_ms: float
    total_latency_ms: float
    groundedness_rate: float
    hallucination_rate: float
    notes: Optional[str] = None
    regression_against: Optional[str] = None
    regression_delta: Dict[str, float] | None = None


def load_qrels(path: Path) -> dict:
    """Load qrels (ground truth) expected in JSON format: {query_id: [doc_ids]}"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkDataset.model_validate(payload)


def _citation_is_relevant(citation: Dict[str, Any], query_spec) -> bool:
    source = (citation.get("source") or "").lower()
    page = citation.get("page")

    if query_spec.relevant_sources and any(pattern.lower() in source for pattern in query_spec.relevant_sources):
        return True

    if query_spec.relevant_pages and page in set(query_spec.relevant_pages):
        return True

    return False


def _relevant_target_count(query_spec) -> int:
    if query_spec.relevant_sources:
        return len(set(query_spec.relevant_sources))
    if query_spec.relevant_pages:
        return len(set(query_spec.relevant_pages))
    return 0


def _precision_at_k(relevance_flags: List[int], k: int) -> float:
    if not relevance_flags:
        return 0.0
    top_flags = relevance_flags[:k]
    return sum(top_flags) / len(top_flags)


def _recall_at_k(relevance_flags: List[int], relevant_count: int, k: int) -> float:
    if relevant_count == 0:
        return 0.0
    return sum(relevance_flags[:k]) / relevant_count


def _mrr(relevance_flags: List[int]) -> float:
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(relevance_flags: List[int], k: int) -> float:
    ranked = relevance_flags[:k]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ranked))
    ideal = sorted(relevance_flags, reverse=True)[:k]
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return (dcg / idcg) if idcg else 0.0


def run_benchmark(
    rag_service,
    dataset: BenchmarkDataset,
    top_k: int = 10,
    benchmark_id: Optional[str] = None,
    baseline: Optional[Dict[str, float]] = None,
    regression_against: Optional[str] = None,
    answer_fn=None,
) -> BenchmarkResult:
    if answer_fn is None:
        from backend.rag.rag_chain import generate_answer as answer_fn

    groundedness_scorer = GroundednessScorer()
    benchmark_started = time.perf_counter()
    query_results: List[BenchmarkQueryResult] = []

    total_retrieval_latency = 0.0
    total_generation_latency = 0.0
    total_reranking_latency = 0.0
    grounded_count = 0
    hallucination_count = 0

    for index, query_spec in enumerate(dataset.queries):
        retrieval_started = time.perf_counter()
        retrieval_result = rag_service.retrieve(query_spec.question)
        if not retrieval_result:
            query_results.append(
                BenchmarkQueryResult(
                    query_id=query_spec.query_id,
                    question=query_spec.question,
                    retrieved_sources=[],
                    relevant_sources=query_spec.relevant_sources,
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    mrr=0.0,
                    ndcg=0.0,
                    groundedness_confidence=0.0,
                    hallucination_rate=1.0,
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    total_latency_ms=0.0,
                    answer_preview=None,
                )
            )
            hallucination_count += 1
            continue

        context, results, retrieval_explanation = retrieval_result
        retrieval_latency = float(getattr(retrieval_explanation, "latency_ms", 0.0))
        if isinstance(retrieval_explanation, dict):
            retrieval_latency = float(retrieval_explanation.get("latency_ms", 0.0))
        if retrieval_latency <= 0.0:
            retrieval_latency = (time.perf_counter() - retrieval_started) * 1000
        total_retrieval_latency += retrieval_latency

        generation_started = time.perf_counter()
        answer_text = answer_fn(context, query_spec.question)
        generation_latency = (time.perf_counter() - generation_started) * 1000
        total_generation_latency += generation_latency

        sources = [
            {
                "source": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:500],
            }
            for doc in results
        ]
        passage_texts = [source.get("preview", "") for source in sources if source.get("preview")]
        groundedness_raw = groundedness_scorer.score_groundedness(answer_text, passage_texts)
        groundedness = GroundednessScore(**groundedness_raw)

        relevance_flags = [1 if _citation_is_relevant(source, query_spec) else 0 for source in sources]
        relevant_count = _relevant_target_count(query_spec)
        precision_at_k = _precision_at_k(relevance_flags, top_k)
        recall_at_k = _recall_at_k(relevance_flags, relevant_count, top_k)
        mrr = _mrr(relevance_flags)
        ndcg = _ndcg_at_k(relevance_flags, top_k)

        grounded_count += 1 if groundedness.is_grounded else 0
        hallucination_count += 1 if groundedness.hallucination_risk == "high" else 0

        total_latency_ms = retrieval_latency + generation_latency
        retrieved_sources = [source.get("source") or "" for source in sources if source.get("source")]
        reranking_latency = 0.0
        retrieval_info = retrieval_explanation.model_dump() if hasattr(retrieval_explanation, "model_dump") else (retrieval_explanation or {})
        if isinstance(retrieval_info, dict):
            reranking_latency = float(retrieval_info.get("stage_timings_ms", {}).get("reranking_ms", 0.0))
        total_reranking_latency += reranking_latency
        query_results.append(
            BenchmarkQueryResult(
                query_id=query_spec.query_id,
                question=query_spec.question,
                retrieved_sources=retrieved_sources,
                relevant_sources=query_spec.relevant_sources,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                mrr=mrr,
                ndcg=ndcg,
                groundedness_confidence=groundedness.confidence,
                hallucination_rate=1.0 if groundedness.hallucination_risk == "high" else 0.0,
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                total_latency_ms=total_latency_ms,
                answer_preview=answer_text[:300],
            )
        )

    if query_results:
        retrieval_metrics = {
            "precision_at_k": sum(result.precision_at_k for result in query_results) / len(query_results),
            "recall_at_k": sum(result.recall_at_k for result in query_results) / len(query_results),
            "map": sum(result.precision_at_k for result in query_results) / len(query_results),
            "mrr": sum(result.mrr for result in query_results) / len(query_results),
            "ndcg": sum(result.ndcg for result in query_results) / len(query_results),
        }
    else:
        retrieval_metrics = {"precision_at_k": 0.0, "recall_at_k": 0.0, "map": 0.0, "mrr": 0.0, "ndcg": 0.0}

    benchmark_id = benchmark_id or f"{dataset.dataset_name}-{int(time.time())}"
    total_latency_ms = (time.perf_counter() - benchmark_started) * 1000
    groundedness_rate = grounded_count / len(dataset.queries) if dataset.queries else 0.0
    hallucination_rate = hallucination_count / len(dataset.queries) if dataset.queries else 0.0

    return BenchmarkResult(
        benchmark_id=benchmark_id,
        dataset_name=dataset.dataset_name,
        timestamp=datetime.now(timezone.utc),
        num_queries=len(dataset.queries),
        retrieval_metrics={
            "precision_at_10": retrieval_metrics.get("precision_at_k", 0.0),
            "recall_at_10": retrieval_metrics.get("recall_at_k", 0.0),
            "map": retrieval_metrics.get("map", 0.0),
            "mrr": retrieval_metrics.get("mrr", 0.0),
            "ndcg": retrieval_metrics.get("ndcg", 0.0),
        },
        retrieval_latency_ms=total_retrieval_latency / max(len(dataset.queries), 1),
        reranking_latency_ms=total_reranking_latency / max(len(dataset.queries), 1) if total_reranking_latency else 0.0,
        total_latency_ms=total_latency_ms,
        groundedness_rate=groundedness_rate,
        hallucination_rate=hallucination_rate,
        notes=None,
        query_results=query_results,
        regression_against=regression_against,
        regression_delta={
            key: retrieval_metrics.get(key, 0.0) - baseline.get(key, 0.0)
            for key in baseline or {}
        },
    )
