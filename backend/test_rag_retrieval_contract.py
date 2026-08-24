from types import SimpleNamespace

from backend.services.rag_service import RagService


class FakeRetrievalService:
    def __init__(self):
        self.calls = []

    def hybrid_retrieve(self, question, top_k):
        self.calls.append((question, top_k))
        return [SimpleNamespace(page_content="context", metadata={"source_file": "doc.pdf", "page": 1})], {
            "expanded_query": None,
            "trace_id": "test-trace",
            "num_dense": top_k,
            "num_sparse": top_k,
            "num_fused": top_k,
            "num_reranked": top_k,
            "origins": [],
            "retrieval_confidence": 0.9,
            "hallucination_risk": "low",
            "stage_timings_ms": {},
        }


def test_retrieve_forwards_explicit_top_k_to_hybrid_retrieval(monkeypatch):
    retrieval = FakeRetrievalService()
    service = RagService(vector_store_service=object(), retrieval_service=retrieval, cache_service=None)

    result = service.retrieve("What is RAG?", top_k=3)

    assert result is not None
    assert retrieval.calls == [("What is RAG?", 3)]
    assert len(result[1]) == 1


def test_retrieve_rejects_invalid_top_k():
    service = RagService(vector_store_service=object(), retrieval_service=FakeRetrievalService(), cache_service=None)

    try:
        service.retrieve("What is RAG?", top_k=0)
    except ValueError as exc:
        assert str(exc) == "top_k must be >= 1"
    else:
        raise AssertionError("retrieve() should reject top_k < 1")
