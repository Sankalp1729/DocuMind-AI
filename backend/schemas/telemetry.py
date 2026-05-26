from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.schemas.retrieval import GroundednessScore, RetrievalOrigin


class RetrievalTimings(BaseModel):
    query_expansion_ms: float
    bm25_retrieval_ms: float
    vector_retrieval_ms: float
    fusion_ms: float
    reranking_ms: float
    compression_ms: float
    total_ms: float


class TelemetryEvent(BaseModel):
    event_type: str  # "retrieval", "ranking", "compression", etc.
    duration_ms: float
    metadata: Dict[str, Any]


class RetrievalStageTiming(BaseModel):
    stage: str
    duration_ms: float = Field(ge=0.0)


class RetrievalTrace(BaseModel):
    trace_id: str
    query: str
    expanded_query: Optional[str] = None
    timestamp: datetime
    total_ms: float = Field(ge=0.0)
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: Optional[str] = None
    num_dense_candidates: int = 0
    num_sparse_candidates: int = 0
    num_fused: int = 0
    num_reranked: int = 0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    origins: List[RetrievalOrigin] = Field(default_factory=list)
    groundedness: Optional[GroundednessScore] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TelemetrySnapshot(BaseModel):
    total_traces: int = 0
    event_averages_ms: Dict[str, float] = Field(default_factory=dict)
    recent_traces: List[RetrievalTrace] = Field(default_factory=list)
