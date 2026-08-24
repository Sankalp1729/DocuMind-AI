from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class BenchmarkQuerySpec(BaseModel):
    query_id: str
    question: str
    relevant_sources: List[str] = Field(default_factory=list)
    relevant_pages: List[int] = Field(default_factory=list)
    answer_references: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("query_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("benchmark query_id/question cannot be empty")
        return value

    @field_validator("relevant_sources", "answer_references")
    @classmethod
    def normalize_text_lists(cls, values: List[str]) -> List[str]:
        normalized = [value.strip() for value in values if value and value.strip()]
        return list(dict.fromkeys(normalized))

    @field_validator("relevant_pages")
    @classmethod
    def validate_pages(cls, values: List[int]) -> List[int]:
        if any(page < 1 for page in values):
            raise ValueError("relevant_pages must contain positive page numbers")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def require_relevance_judgment(self) -> "BenchmarkQuerySpec":
        if not self.relevant_sources and not self.relevant_pages:
            raise ValueError(
                "each benchmark query needs at least one relevance judgment "
                "in relevant_sources or relevant_pages"
            )
        return self


class BenchmarkDataset(BaseModel):
    dataset_name: str
    description: str
    version: str = "1.0"
    queries: List[BenchmarkQuerySpec]

    @field_validator("dataset_name", "description", "version")
    @classmethod
    def validate_dataset_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("benchmark dataset metadata cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_query_ids(self) -> "BenchmarkDataset":
        if not self.queries:
            raise ValueError("benchmark dataset must contain at least one query")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("benchmark query_id values must be unique")
        return self


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
    top_k: int = Field(ge=1)
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
