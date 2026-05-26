from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BenchmarkQuerySpec(BaseModel):
    query_id: str
    question: str
    relevant_sources: List[str] = Field(default_factory=list)
    relevant_pages: List[int] = Field(default_factory=list)
    answer_references: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class BenchmarkDataset(BaseModel):
    dataset_name: str
    description: str
    version: str = "1.0"
    queries: List[BenchmarkQuerySpec]


class BenchmarkQueryResult(BaseModel):
    query_id: str
    question: str
    retrieved_sources: List[str]
    relevant_sources: List[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    groundedness_confidence: float
    hallucination_rate: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    answer_preview: Optional[str] = None


class BenchmarkResult(BaseModel):
    benchmark_id: str
    dataset_name: str
    timestamp: datetime
    num_queries: int
    retrieval_metrics: Dict[str, float]
    retrieval_latency_ms: float
    reranking_latency_ms: Optional[float] = None
    total_latency_ms: float
    groundedness_rate: float
    hallucination_rate: float
    notes: Optional[str] = None
    query_results: List[BenchmarkQueryResult] = Field(default_factory=list)
    regression_against: Optional[str] = None
    regression_delta: Dict[str, float] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    map: float
    mrr: float
    ndcg: float


class EvaluationRun(BaseModel):
    run_id: str
    dataset: str
    metrics: EvaluationMetrics
    timestamp: datetime
    duration_seconds: float


class LeaderboardEntry(BaseModel):
    benchmark_id: str
    dataset_name: str
    timestamp: datetime
    composite_score: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg: float
    groundedness_rate: float
    hallucination_rate: float
    total_latency_ms: float


class EvaluationHistoryEntry(BaseModel):
    benchmark_id: str
    dataset_name: str
    timestamp: datetime
    composite_score: float
    metrics: Dict[str, float]
    regression_delta: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None
