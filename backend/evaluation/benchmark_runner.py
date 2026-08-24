from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    source_targets = {source.lower() for source in query_spec.relevant_sources}
    page_targets = {page for page in query_spec.relevant_pages}
    return len(source_targets) if source_targets else len(page_targets)


def _precision_at_k(relevance_flags: List[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    top_flags = relevance_flags[:k]
    return sum(top_flags) / len(top_flags) if top_flags else 0.0


def _recall_at_k(relevance_flags: List[int], relevant_count: int, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if relevant_count <= 0:
        return 0.0
    return min(1.0, sum(relevance_flags[:k]) / relevant_count)


def _average_precision(relevance_flags: List[int], relevant_count: int) -> float:
    if not relevance_flags or relevant_count <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            hits += 1
            score += hits / rank
    return min(1.0, score / relevant_count) if hits else 0.0


def _mrr(relevance_flags: List[int]) -> float:
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(relevance_flags: List[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked = relevance_flags[:k]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ranked))
    ideal = sorted(relevance_flags, reverse=True)[:k]
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _retrieve_for_benchmark(rag_service, question: str, top_k: int):
    """Use explicit K on real services; support older lightweight test doubles."""
    try:
        return rag_service.retrieve(question, top_k=top_k)
    except TypeError as exc:
        if "top_k" not in str(exc):
            raise
        return rag_service.retrieve(question)


def run_benchmark(
    rag_service,
    dataset: BenchmarkDataset,
    top_k: int = 10,
    benchmark_id: Optional[str] = None,
    baseline: Optional[Dict[str, float]] = None,
    regression_against: Optional[str] = None,
    answer_fn=None,
) -> BenchmarkResult:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if answer_fn is None:
        from backend.rag.rag_chain import generate_answer as answer_fn

    groundedness_scorer = GroundednessScorer()
    benchmark_started = time.perf_counter()
    query_results: List[BenchmarkQueryResult] = []
    average_precisions: List[float] = []
    total_retrieval_latency = total_generation_latency = total_reranking_latency = 0.0
    grounded_count = hallucination_count = 0

    for query_spec in dataset.queries:
        retrieval_started = time.perf_counter()
        retrieval_result = _retrieve_for_benchmark(rag_service, query_spec.question, top_k)
        if not retrieval_result:
            query_results.append(BenchmarkQueryResult(
                query_id=query_spec.query_id, question=query_spec.question, retrieved_sources=[],
                relevant_sources=query_spec.relevant_sources, precision_at_k=0.0, recall_at_k=0.0,
                mrr=0.0, ndcg=0.0, groundedness_confidence=0.0, hallucination_rate=1.0,
                retrieval_latency_ms=0.0, generation_latency_ms=0.0, total_latency_ms=0.0,
                answer_preview=None,
            ))
            average_precisions.append(0.0)
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
            {"source": doc.metadata.get("source_file"), "page": doc.metadata.get("page"), "preview": doc.page_content[:500]}
            for doc in results
        ]
        passage_texts = [source["preview"] for source in sources if source.get("preview")]
        groundedness = GroundednessScore(**groundedness_scorer.score_groundedness(answer_text, passage_texts))

        relevance_flags = [1 if _citation_is_relevant(source, query_spec) else 0 for source in sources]
        relevant_count = _relevant_target_count(query_spec)
        precision_at_k = _precision_at_k(relevance_flags, top_k)
        recall_at_k = _recall_at_k(relevance_flags, relevant_count, top_k)
        average_precisions.append(_average_precision(relevance_flags[:top_k], relevant_count))
        mrr = _mrr(relevance_flags[:top_k])
        ndcg = _ndcg_at_k(relevance_flags, top_k)
        grounded_count += int(groundedness.is_grounded)
        hallucination_count += int(groundedness.hallucination_risk == "high")

        retrieval_info = retrieval_explanation.model_dump() if hasattr(retrieval_explanation, "model_dump") else (retrieval_explanation or {})
        reranking_latency = float(retrieval_info.get("stage_timings_ms", {}).get("reranking_ms", 0.0)) if isinstance(retrieval_info, dict) else 0.0
        total_reranking_latency += reranking_latency
        query_results.append(BenchmarkQueryResult(
            query_id=query_spec.query_id, question=query_spec.question,
            retrieved_sources=[source["source"] for source in sources if source.get("source")],
            relevant_sources=query_spec.relevant_sources, precision_at_k=precision_at_k,
            recall_at_k=recall_at_k, mrr=mrr, ndcg=ndcg,
            groundedness_confidence=groundedness.confidence,
            hallucination_rate=1.0 if groundedness.hallucination_risk == "high" else 0.0,
            retrieval_latency_ms=retrieval_latency, generation_latency_ms=generation_latency,
            total_latency_ms=retrieval_latency + generation_latency, answer_preview=answer_text[:300],
        ))

    count = len(query_results)
    metric_key_precision = f"precision_at_{top_k}"
    metric_key_recall = f"recall_at_{top_k}"
    retrieval_metrics = {
        metric_key_precision: sum(r.precision_at_k for r in query_results) / count if count else 0.0,
        metric_key_recall: sum(r.recall_at_k for r in query_results) / count if count else 0.0,
        "map": sum(average_precisions) / len(average_precisions) if average_precisions else 0.0,
        "mrr": sum(r.mrr for r in query_results) / count if count else 0.0,
        "ndcg": sum(r.ndcg for r in query_results) / count if count else 0.0,
    }
    retrieval_metrics["precision_at_10"] = retrieval_metrics[metric_key_precision]
    retrieval_metrics["recall_at_10"] = retrieval_metrics[metric_key_recall]

    benchmark_id = benchmark_id or f"{dataset.dataset_name}-{int(time.time())}"
    num_queries = len(dataset.queries)
    return BenchmarkResult(
        benchmark_id=benchmark_id, dataset_name=dataset.dataset_name, timestamp=datetime.now(timezone.utc),
        num_queries=num_queries, retrieval_metrics=retrieval_metrics,
        retrieval_latency_ms=total_retrieval_latency / max(num_queries, 1),
        reranking_latency_ms=total_reranking_latency / max(num_queries, 1) if total_reranking_latency else 0.0,
        total_latency_ms=(time.perf_counter() - benchmark_started) * 1000,
        groundedness_rate=grounded_count / num_queries if num_queries else 0.0,
        hallucination_rate=hallucination_count / num_queries if num_queries else 0.0,
        notes=None, query_results=query_results, regression_against=regression_against,
        regression_delta={key: retrieval_metrics.get(key, 0.0) - baseline.get(key, 0.0) for key in baseline or {}},
    )
