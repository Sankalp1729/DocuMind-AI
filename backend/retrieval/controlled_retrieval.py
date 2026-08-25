from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from backend.retrieval.retrieval_fusion import reciprocal_rank_fusion
from backend.retrieval.strategy import RetrievalStrategy


class ControlledRetrieval:
    """Run retrieval experiments without changing the production RAG path.

    This adapter deliberately separates fusion from reranking so benchmark runs
    can compare the two stages independently.
    """

    def __init__(self, retrieval_service):
        self.service = retrieval_service

    def retrieve(self, query: str, top_k: int, strategy: RetrievalStrategy) -> Tuple[List[Document], Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        strategy = RetrievalStrategy(strategy)

        if strategy == RetrievalStrategy.DENSE:
            hits = self.service.vector_retriever.retrieve(query, top_k=top_k)
            return [doc for _id, _score, doc in hits], self._info(strategy, len(hits), 0, 0, False)

        self.service._ensure_bm25()
        texts = self.service._extract_documents_from_faiss()
        if not texts or self.service.bm25 is None:
            return [], self._info(strategy, 0, 0, 0, strategy == RetrievalStrategy.HYBRID_RERANKED)

        candidate_k = min(max(top_k * 2, top_k), len(texts))
        sparse = self.service.bm25.retrieve(query, top_k=candidate_k)
        dense = self.service.vector_retriever.retrieve(query, top_k=candidate_k)

        if strategy == RetrievalStrategy.BM25:
            docs = self._docs_from_indices([int(i) for i, _score in sparse[:top_k]])
            return docs, self._info(strategy, 0, len(sparse), 0, False)

        dense_ranked = [(str(i), float(score)) for i, score, _doc in dense]
        sparse_ranked = [(str(i), float(score)) for i, score in sparse]
        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=60)
        fused_ids = [int(i) for i, _score in fused if str(i).isdigit()]

        if strategy == RetrievalStrategy.HYBRID:
            docs = self._docs_from_indices(fused_ids[:top_k])
            return docs, self._info(strategy, len(dense), len(sparse), len(fused), False)

        candidate_ids = fused_ids[: max(top_k * 2, top_k)]
        candidate_docs = self._docs_from_indices(candidate_ids)
        reranker = self.service.reranker
        if reranker is None:
            # The strategy remains explicit: no silent fallback to the production
            # hybrid path. Returning the fused candidates makes the missing
            # optional dependency observable to the benchmark.
            return candidate_docs[:top_k], self._info(strategy, len(dense), len(sparse), len(fused), False, reranker_available=False)

        reranked = reranker.rerank(query, [doc.page_content for doc in candidate_docs])
        final_docs = [candidate_docs[pos] for pos, _score in reranked[:top_k] if 0 <= pos < len(candidate_docs)]
        return final_docs, self._info(strategy, len(dense), len(sparse), len(fused), True)

    def _docs_from_indices(self, indices: List[int]) -> List[Document]:
        store = self.service.vs_service.get_vector_store()
        if not hasattr(store, "docstore"):
            return []
        docs: List[Document] = []
        for index in indices:
            doc_id = self.service._document_indices_to_ids.get(index)
            if not doc_id:
                continue
            doc = store.docstore.search(doc_id)
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _info(strategy: RetrievalStrategy, dense: int, sparse: int, fused: int, reranked: bool, reranker_available: bool = True) -> Dict[str, Any]:
        return {
            "strategy": strategy.value,
            "reranker": reranked,
            "reranker_available": reranker_available,
            "num_dense": dense,
            "num_sparse": sparse,
            "num_fused": fused,
            "num_reranked": reranked,
        }
