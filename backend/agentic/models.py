from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentStep:
    step_id: str
    kind: str
    query: str
    rationale: str | None = None
    tool_name: str | None = None
    status: str = "planned"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentPlan:
    question: str
    intent: str
    steps: list[AgentStep] = field(default_factory=list)
    needs_web: bool = False
    needs_memory: bool = False
    needs_reflection: bool = True
    complexity: str = "medium"


@dataclass(slots=True)
class ToolObservation:
    tool_name: str
    query: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgenticRetrievalResult:
    answer: str
    context: str
    sources: list[dict[str, Any]]
    retrieval_explanation: dict[str, Any]
    plan: AgentPlan
    observations: list[ToolObservation] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
