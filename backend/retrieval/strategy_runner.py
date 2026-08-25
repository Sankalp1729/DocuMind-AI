from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from backend.retrieval.strategy import RetrievalStrategy


class RetrievalStrategyRunner:
    """Execute an explicit retrieval strategy against the existing retrieval service."""

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service

    def retrieve(self, query: str, top_k: int, strategy: RetrievalStrategy) -> Tuple[List[Document], Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        strategy = RetrievalStrategy(strategy)
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_RERANKED):
            # The current production hybrid path already performs its configured
            # reranking. Keep the adapter honest rather than passing unsupported
            # flags into the production service.
            docs, info = self.retrieval_service.hybrid_retrieve(query, top_k=top_k)
            info = dict(info)
            info["strategy"] = strategy.value
            info["reranker"] = strategy == RetrievalStrategy.HYBRID_RERANKED
            return docs, info

        if strategy == RetrievalStrategy.DENSE:
            hits = self.retrieval_service.vector_retriever.retrieve(query, top_k=top_k)
            docs = [doc for _doc_id, _score, doc in hits]
            return docs, {
                "strategy": strategy.value,
                "reranker": False,
                "num_dense": len(docs),
                "num_sparse": 0,
                "num_fused": 0,
                "num_reranked": len(docs),
                "origins": [],
                "stage_timings_ms": {},
            }

        self.retrieval_service._ensure_bm25()
        texts = self.retrieval_service._extract_documents_from_faiss()
        if not texts or self.retrieval_service.bm25 is None:
            return [], {
                "strategy": strategy.value,
                "reranker": False,
                "num_dense": 0,
                "num_sparse": 0,
                "num_fused": 0,
                "num_reranked": 0,
                "origins": [],
                "stage_timings_ms": {},
            }

        hits = self.retrieval_service.bm25.retrieve(query, top_k=top_k)
        vector_store = self.retrieval_service.vs_service.get_vector_store()
        docs: List[Document] = []
        if hasattr(vector_store, "docstore"):
            for index, _score in hits:
                doc_id = self.retrieval_service._document_indices_to_ids.get(index)
                if doc_id:
                    doc = vector_store.docstore.search(doc_id)
                    if doc:
                        docs.append(doc)

        return docs, {
            "strategy": strategy.value,
            "reranker": False,
            "num_dense": 0,
            "num_sparse": len(docs),
            "num_fused": 0,
            "num_reranked": len(docs),
            "origins": [],
            "stage_timings_ms": {},
        }
