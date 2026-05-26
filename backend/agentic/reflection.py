from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReflectionResult:
    approved: bool
    critique: str
    revision_hint: str | None = None


class SelfReflectionLoop:
    """Simple evidence-based reflection loop.

    The first pass is judged against heuristics so the agent can optionally do one revision pass.
    """

    def reflect(self, question: str, answer: str, evidence: str, citations: list[dict[str, Any]]) -> ReflectionResult:
        evidence_len = len(evidence.strip())
        has_citations = bool(citations)
        if evidence_len < 40 or not has_citations:
            return ReflectionResult(
                approved=False,
                critique="Insufficient grounded evidence or citations.",
                revision_hint="Use more retrieval evidence and cite sources explicitly.",
            )
        if len(answer.strip()) < 20:
            return ReflectionResult(
                approved=False,
                critique="Answer is too short to be useful.",
                revision_hint="Expand the answer with the strongest retrieved facts.",
            )
        return ReflectionResult(approved=True, critique="Answer is sufficiently grounded.")
