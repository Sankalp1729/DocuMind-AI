from __future__ import annotations

from backend.agentic.memory import MemoryEnhancedRetrieval
from backend.agentic.planner import QueryPlanner
from backend.agentic.reflection import SelfReflectionLoop


def test_query_planner_creates_multi_step_plan():
    planner = QueryPlanner()
    plan = planner._heuristic_plan("Compare ingestion and retrieval, then explain latest scaling tradeoffs")
    assert plan.steps
    assert any(step.kind == "retrieve" for step in plan.steps)
    assert plan.needs_web is True
    assert plan.complexity in {"medium", "high"}


def test_memory_recall_and_remember(tmp_path):
    memory = MemoryEnhancedRetrieval(storage_dir=tmp_path)
    memory.remember("How do I scale retrieval?", "Use sharding and collection-level indexing.")
    hits = memory.recall("scale retrieval")
    assert hits
    assert "scale retrieval" in hits[0].query.lower()


def test_self_reflection_flags_weak_evidence():
    reflector = SelfReflectionLoop()
    result = reflector.reflect("question", "short", "", [])
    assert result.approved is False
    assert result.revision_hint is not None
