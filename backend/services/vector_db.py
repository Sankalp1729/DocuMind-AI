from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


class VectorDB:
    """Abstract vector DB interface."""

    def upsert(self, collection: str, vectors: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        raise NotImplementedError()

    def search(self, collection: str, query_vector: List[float] | None = None, query: str | None = None, top_k: int = 10, filter: dict | None = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return list of tuples (id, score, metadata/document dict).

        Either `query_vector` or `query` (text) can be provided depending on backend.
        """
        raise NotImplementedError()

    def create_collection(self, collection: str, vector_size: int, distance: str = "Cosine") -> None:
        raise NotImplementedError()

    def delete_collection(self, collection: str) -> None:
        raise NotImplementedError()

    def collection_exists(self, collection: str) -> bool:
        raise NotImplementedError()

    def list_documents(self, collection: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return list of (id, text, payload) for all documents in a collection.

        This is used to build sparse indexes (BM25) from the persistent vector store.
        """
        raise NotImplementedError()
