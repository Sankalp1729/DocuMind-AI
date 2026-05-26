from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from backend.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from backend.agentic.models import AgentPlan, AgentStep

logger = logging.getLogger(__name__)


class QueryPlanner:
    """Create a lightweight execution plan for agentic RAG.

    Uses heuristics first and can optionally ask the LLM for a structured plan.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
        )
        self._prompt = PromptTemplate(
            template=(
                "You are a retrieval planner. Return JSON only with keys: intent, complexity, needs_web, needs_memory, steps. "
                "Each step must have kind, query, and rationale. Keep it concise. Question: {question}"
            ),
            input_variables=["question"],
        )

    def _heuristic_plan(self, question: str) -> AgentPlan:
        lower = question.lower()
        needs_web = any(token in lower for token in ["latest", "current", "news", "web", "internet", "today", "recent"])
        needs_memory = any(token in lower for token in ["previous", "earlier", "remember", "again", "compare"])
        complexity = "high" if any(token in lower for token in ["compare", "analyze", "plan", "why", "how", "multi-step"]) else "medium"

        parts = [segment.strip() for segment in re.split(r"\b(?:and then|then|also|compare|before|after)\b", question) if segment.strip()]
        if not parts:
            parts = [question.strip()]

        steps = []
        for index, part in enumerate(parts, start=1):
            steps.append(
                AgentStep(
                    step_id=f"step-{index}",
                    kind="retrieve",
                    query=part,
                    rationale="Retrieve evidence for the sub-question",
                )
            )

        if needs_web:
            steps.append(
                AgentStep(
                    step_id=f"step-{len(steps)+1}",
                    kind="web_search",
                    query=question,
                    rationale="Augment with web context for recency or external facts",
                    tool_name="web_search",
                )
            )

        return AgentPlan(
            question=question,
            intent="Answer the user's question using document evidence and optional web context.",
            steps=steps,
            needs_web=needs_web,
            needs_memory=needs_memory,
            needs_reflection=True,
            complexity=complexity,
        )

    def plan(self, question: str) -> AgentPlan:
        plan = self._heuristic_plan(question)
        try:
            response = (self._prompt | self.llm).invoke({"question": question})
            payload = json.loads(response.content)
            steps = []
            for index, item in enumerate(payload.get("steps", []), start=1):
                steps.append(
                    AgentStep(
                        step_id=f"step-{index}",
                        kind=item.get("kind", "retrieve"),
                        query=item.get("query", question),
                        rationale=item.get("rationale"),
                        tool_name=item.get("tool_name"),
                    )
                )
            if steps:
                plan.steps = steps
            plan.intent = payload.get("intent", plan.intent)
            plan.complexity = payload.get("complexity", plan.complexity)
            plan.needs_web = bool(payload.get("needs_web", plan.needs_web))
            plan.needs_memory = bool(payload.get("needs_memory", plan.needs_memory))
        except Exception as exc:
            logger.debug("Falling back to heuristic plan: %s", exc)
        return plan
