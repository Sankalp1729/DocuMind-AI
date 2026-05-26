from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from backend.agentic.memory import MemoryEnhancedRetrieval
from backend.agentic.models import AgenticRetrievalResult, ToolObservation
from backend.agentic.planner import QueryPlanner
from backend.agentic.reflection import SelfReflectionLoop
from backend.agentic.tools import RetrievalTool, WebFetchTool, WebSearchTool
from backend.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
if TYPE_CHECKING:
    from backend.services.rag_service import RagService


logger = logging.getLogger(__name__)


class AgenticRagService:
    """Agentic RAG orchestration with planning, tool use, memory, and reflection.

    The service stays modular by delegating retrieval to RagService and external facts to web tools.
    """

    def __init__(self, rag_service: "RagService", memory_store: MemoryEnhancedRetrieval | None = None):
        self.rag_service = rag_service
        self.planner = QueryPlanner()
        self.retrieval_tool = RetrievalTool(rag_service)
        self.web_search_tool = WebSearchTool()
        self.web_fetch_tool = WebFetchTool()
        self.memory = memory_store or MemoryEnhancedRetrieval()
        self.reflector = SelfReflectionLoop()
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
        self.answer_prompt = PromptTemplate(
            template=(
                "You are an autonomous retrieval agent. Use the context and observations to answer the question. "
                "Explain your reasoning briefly, cite sources, and avoid unsupported claims.\n\n"
                "Question: {question}\n\n"
                "Plan: {plan}\n\n"
                "Observations:\n{observations}\n\n"
                "Memory:\n{memory}\n\n"
                "Answer with citations:" 
            ),
            input_variables=["question", "plan", "observations", "memory"],
        )

    def _format_observations(self, observations: list[ToolObservation]) -> str:
        chunks = []
        for obs in observations:
            chunks.append(f"[{obs.tool_name}] {obs.query}\n{obs.content}")
        return "\n\n".join(chunks)

    def _format_memory(self, memories) -> str:
        if not memories:
            return "None"
        return "\n".join(f"- Q: {item.query}\n  A: {item.answer[:500]}" for item in memories)

    def answer(self, question: str) -> dict[str, Any] | None:
        plan = self.planner.plan(question)
        observations: list[ToolObservation] = []
        citations: list[dict[str, Any]] = []

        memory_hits = self.memory.recall(question) if plan.needs_memory else []
        if memory_hits:
            observations.append(
                ToolObservation(
                    tool_name="memory_recall",
                    query=question,
                    content="\n".join(f"Q: {item.query}\nA: {item.answer}" for item in memory_hits),
                    metadata={"hits": len(memory_hits)},
                )
            )

        first_pass_context = []
        for step in plan.steps:
            if step.kind == "web_search":
                result = self.web_search_tool.search(step.query)
                observations.append(result.observation)
                citations.extend(result.citations)
                continue

            if step.kind == "web_fetch":
                result = self.web_fetch_tool.fetch(step.query)
                observations.append(result.observation)
                citations.extend(result.citations)
                continue

            result = self.retrieval_tool.run(step.query)
            observations.append(result.observation)
            citations.extend(result.citations)
            first_pass_context.append(result.observation.content)

        if plan.needs_web and not any(obs.tool_name == "web_search" for obs in observations):
            result = self.web_search_tool.search(question)
            observations.append(result.observation)
            citations.extend(result.citations)

        memory_text = self._format_memory(memory_hits)
        obs_text = self._format_observations(observations)
        chain_input = {
            "question": question,
            "plan": plan.intent + " | steps=" + "; ".join(f"{s.kind}:{s.query}" for s in plan.steps),
            "observations": obs_text,
            "memory": memory_text,
        }

        response = (self.answer_prompt | self.llm).invoke(chain_input)
        answer = response.content

        reflection = self.reflector.reflect(question, answer, obs_text, citations)
        reflections = [reflection.critique]
        if not reflection.approved and reflection.revision_hint:
            revision_prompt = PromptTemplate(
                template=(
                    "Revise the answer using the critique and evidence.\n\n"
                    "Question: {question}\n\n"
                    "Previous answer: {answer}\n\n"
                    "Critique: {critique}\n\n"
                    "Evidence:\n{evidence}\n\n"
                    "Improved answer with citations:"
                ),
                input_variables=["question", "answer", "critique", "evidence"],
            )
            revised = (revision_prompt | self.llm).invoke(
                {
                    "question": question,
                    "answer": answer,
                    "critique": reflection.revision_hint,
                    "evidence": obs_text,
                }
            )
            answer = revised.content
            reflections.append("Revised after self-reflection loop")

        self.memory.remember(question, answer, metadata={"plan": asdict(plan), "reflection": reflections[-1] if reflections else None})

        return {
            "question": question,
            "answer": answer,
            "plan": plan,
            "observations": observations,
            "reflections": reflections,
            "citations": citations,
        }
