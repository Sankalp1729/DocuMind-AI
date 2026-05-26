from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalOrigin(BaseModel):
    source: str
    origin: str  # "bm25", "vector", or "fused"
    rank: Optional[int] = None
    document_id: Optional[str] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    confidence_reasoning: Optional[str] = None
    hallucination_risk: Optional[str] = None


class RetrievalExplanation(BaseModel):
    query: str
    expanded_query: Optional[str] = None
    trace_id: Optional[str] = None
    num_dense_candidates: int
    num_sparse_candidates: int
    num_fused: int
    num_reranked: int
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: Optional[str] = None
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    origins: List[RetrievalOrigin]
    latency_ms: float


class GroundednessScore(BaseModel):
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: str  # "low", "medium", "high"
    reasoning: str
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    question_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_signals: List[str] = Field(default_factory=list)
    support_summary: Optional[str] = None


class EnhancedCitation(BaseModel):
    source: str
    page: Optional[int] = None
    preview: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    origin: RetrievalOrigin
    groundedness: Optional[GroundednessScore] = None
