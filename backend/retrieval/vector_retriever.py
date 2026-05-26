from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_core.documents import Document

from backend.rag.vector_store import get_embeddings
from backend.services.vector_store_service import VectorStoreService


class VectorRetriever:
    """Backend-agnostic vector retriever supporting FAISS and remote vector DBs (Qdrant).

    Supports optional metadata filtering for retrieval.
    """

    def __init__(self, vector_store_service: VectorStoreService):
        self.vs = vector_store_service

    def retrieve(self, query: str, top_k: int = 10, metadata_filter: dict | None = None) -> List[Tuple[str, float, Document]]:
        vs = self.vs.get_vector_store()
        if vs is None:
            return []

        # Qdrant backend returns tuple (backend, collection_name)
        if isinstance(vs, tuple) and len(vs) == 2:
            backend, collection = vs
            # embed query text
            embedder = get_embeddings()
            qvec = embedder.embed_query(query)
            results = backend.search(collection, query_vector=qvec, top_k=top_k, filter=metadata_filter)
            ranked: List[Tuple[str, float, Document]] = []
            for _id, score, payload in results:
                doc = Document(page_content=payload.get("text", payload.get("content", "")), metadata=payload)
                ranked.append((_id, float(score), doc))
            return ranked

        # FAISS / langchain vectorstore
        vector_store = vs
        docs = vector_store.similarity_search_with_score(query, k=top_k)
        index_to_docstore_id = getattr(vector_store, "index_to_docstore_id", {})
        docstore = getattr(vector_store, "docstore", None)

        ranked: List[Tuple[str, float, Document]] = []
        for doc, score in docs:
            doc_id = None
            if docstore is not None:
                for index, candidate_doc_id in index_to_docstore_id.items():
                    candidate_doc = docstore.search(candidate_doc_id)
                    if candidate_doc and candidate_doc.page_content == doc.page_content and candidate_doc.metadata == doc.metadata:
                        doc_id = str(index)
                        break
            ranked.append((doc_id or doc.metadata.get("chunk_id") or doc.metadata.get("source_file") or str(len(ranked)), float(score), doc))

        return ranked
