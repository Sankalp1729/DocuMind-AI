import pytest
from pydantic import ValidationError

from backend.schemas.evaluation import BenchmarkDataset, BenchmarkQuerySpec


def _query(**overrides):
    payload = {
        "query_id": "q1",
        "question": "What does the system retrieve?",
        "relevant_sources": ["retrieval.md"],
        "relevant_pages": [1],
    }
    payload.update(overrides)
    return payload


def test_query_normalizes_text_and_deduplicates_judgments() -> None:
    query = BenchmarkQuerySpec(
        **_query(
            query_id=" q1 ",
            question=" What does the system retrieve? ",
            relevant_sources=[" retrieval.md ", "retrieval.md", ""],
            relevant_pages=[1, 1],
        )
    )

    assert query.query_id == "q1"
    assert query.relevant_sources == ["retrieval.md"]
    assert query.relevant_pages == [1]


def test_query_requires_relevance_judgment() -> None:
    with pytest.raises(ValidationError, match="at least one relevance judgment"):
        BenchmarkQuerySpec(
            query_id="q1",
            question="What does the system retrieve?",
        )


def test_query_rejects_invalid_pages() -> None:
    with pytest.raises(ValidationError, match="positive page numbers"):
        BenchmarkQuerySpec(**_query(relevant_pages=[0]))


def test_dataset_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValidationError, match="query_id values must be unique"):
        BenchmarkDataset(
            dataset_name="demo",
            description="A benchmark dataset",
            queries=[_query(query_id="q1"), _query(query_id="q1")],
        )


def test_dataset_requires_at_least_one_query() -> None:
    with pytest.raises(ValidationError, match="at least one query"):
        BenchmarkDataset(
            dataset_name="demo",
            description="A benchmark dataset",
            queries=[],
        )
