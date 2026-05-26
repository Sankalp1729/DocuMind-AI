from __future__ import annotations

from typing import Literal, Dict, Any

from pydantic import BaseModel, Field


StreamingEventType = Literal["start", "token", "complete", "error"]


class StreamingRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class StreamingCitation(BaseModel):
    source: str | None = None
    page: int | None = None
    preview: str | None = None


class StreamingEvent(BaseModel):
    event: StreamingEventType
    question: str | None = None
    token: str | None = None
    answer: str | None = None
    sources: list[StreamingCitation] | None = None
    retrieval_explanation: Dict[str, Any] | None = None
    groundedness: Dict[str, Any] | None = None
    latency_ms: int | None = None
    error: str | None = None