from __future__ import annotations

from datetime import datetime, timezone

from backend.evaluation.groundedness import GroundednessScorer
from backend.schemas.retrieval import GroundednessScore, RetrievalOrigin
from backend.schemas.telemetry import RetrievalTrace
from backend.services.telemetry_service import TelemetryService


def test_groundedness_reports_hallucination_signals() -> None:
    scorer = GroundednessScorer()
    result = scorer.score_groundedness(
        "I cannot verify the exact value, but it may be 42.",
        ["The document states that the answer is derived from the policy."],
    )

    assert result["is_grounded"] is False
    assert result["hallucination_risk"] == "high"
    assert "explicit_uncertainty_marker" in result["hallucination_signals"]
    assert result["support_summary"]


def test_telemetry_service_persists_retrieval_traces(tmp_path) -> None:
    service = TelemetryService(storage_dir=tmp_path)
    trace = RetrievalTrace(
        trace_id="trace-1",
        query="what is retrieval explainability",
        expanded_query="what is retrieval explainability diagnostics",
        timestamp=datetime.now(timezone.utc),
        total_ms=12.5,
        retrieval_confidence=0.88,
        hallucination_risk="low",
        num_dense_candidates=3,
        num_sparse_candidates=2,
        num_fused=4,
        num_reranked=2,
        stage_timings_ms={"vector_retrieval_ms": 3.2, "reranking_ms": 1.1},
        origins=[
            RetrievalOrigin(
                source="doc.pdf",
                origin="fused",
                rank=0,
                document_id="0",
                bm25_score=0.42,
                vector_score=0.18,
                rrf_score=0.03,
                reranker_score=0.91,
                confidence=0.88,
                score_breakdown={"bm25_normalized": 0.9},
                confidence_reasoning="Strong agreement",
                hallucination_risk="low",
            )
        ],
        groundedness=GroundednessScore(
            is_grounded=True,
            confidence=0.91,
            hallucination_risk="low",
            reasoning="High lexical overlap",
            evidence_coverage=0.82,
            question_alignment=0.77,
            hallucination_signals=[],
            support_summary="Supported by the retrieved passage",
        ),
    )

    service.record_retrieval_trace(trace)

    traces = service.list_traces()
    summary = service.get_summary()

    assert traces[-1].trace_id == "trace-1"
    assert summary.total_traces == 1
    assert summary.recent_traces[-1].query == "what is retrieval explainability"
    assert (tmp_path / "retrieval_traces.jsonl").exists()
