from types import SimpleNamespace

import pytest

from backend.retrieval.controlled_retrieval import ControlledRetrieval
from backend.retrieval.strategy import RetrievalStrategy


class FakeBM25:
    def retrieve(self, query, top_k=10):
        return [(0, 2.0), (1, 1.0)][:top_k]


class FakeVector:
    def retrieve(self, query, top_k=10):
        docs = [
            (0, 0.9, SimpleNamespace(page_content="dense zero")),
            (1, 0.8, SimpleNamespace(page_content="dense one")),
        ]
        return docs[:top_k]


class FakeReranker:
    def rerank(self, query, texts):
        return list(reversed([(i, float(i)) for i in range(len(texts))]))


class FakeStore:
    def __init__(self):
        self.docstore = {
            "doc0": SimpleNamespace(page_content="zero"),
            "doc1": SimpleNamespace(page_content="one"),
        }

    def search(self, doc_id):
        return self.docstore.get(doc_id)


class FakeService:
    def __init__(self, reranker=True):
        self.bm25 = FakeBM25()
        self.vector_retriever = FakeVector()
        self._document_indices_to_ids = {0: "doc0", 1: "doc1"}
        self.vs_service = SimpleNamespace(get_vector_store=lambda: SimpleNamespace(docstore=FakeStore()))
        self._reranker = FakeReranker() if reranker else None

    def _ensure_bm25(self):
        return None

    def _extract_documents_from_faiss(self):
        return ["zero", "one"]

    @property
    def reranker(self):
        return self._reranker


def test_hybrid_does_not_call_reranker(monkeypatch):
    service = FakeService()
    calls = {"count": 0}

    def fail(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("reranker must not run for hybrid")

    service._reranker.rerank = fail
    docs, info = ControlledRetrieval(service).retrieve("query", 1, RetrievalStrategy.HYBRID)

    assert len(docs) == 1
    assert info["strategy"] == "hybrid"
    assert info["reranker"] is False
    assert calls["count"] == 0


def test_hybrid_reranked_calls_cross_encoder():
    service = FakeService()
    docs, info = ControlledRetrieval(service).retrieve("query", 1, RetrievalStrategy.HYBRID_RERANKED)

    assert len(docs) == 1
    assert info["strategy"] == "hybrid_reranked"
    assert info["reranker"] is True


def test_missing_reranker_is_observable():
    service = FakeService(reranker=False)
    docs, info = ControlledRetrieval(service).retrieve("query", 1, RetrievalStrategy.HYBRID_RERANKED)

    assert len(docs) == 1
    assert info["reranker"] is False
    assert info["reranker_available"] is False


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ControlledRetrieval(FakeService()).retrieve("query", 0, RetrievalStrategy.DENSE)
