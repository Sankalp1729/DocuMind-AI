from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from backend.services.vector_db import VectorDB


logger = logging.getLogger(__name__)


class QdrantBackend(VectorDB):
    def __init__(self, url: str = "localhost", port: int = 6333, prefer_grpc: bool = False):
        self.client = QdrantClient(url=url, port=port, prefer_grpc=prefer_grpc)

    def create_collection(self, collection: str, vector_size: int, distance: str = "Cosine") -> None:
        if self.client.get_collection(collection_name=collection) is not None:
            return
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE if distance.lower() == "cosine" else rest.Distance.EUCLID),
        )

    def upsert(self, collection: str, vectors: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        points = []
        for _id, vector, payload in vectors:
            points.append(rest.PointStruct(id=_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query_vector: List[float] | None = None, query: str | None = None, top_k: int = 10, filter: dict | None = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        if query_vector is None:
            raise ValueError("Qdrant backend requires a numeric query_vector")

        qfilter = None
        if filter:
            qfilter = rest.Filter(**filter)

        hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k, query_filter=qfilter)
        results = []
        for hit in hits:
            results.append((str(hit.id), float(hit.score), hit.payload or {}))
        return results

    def delete_collection(self, collection: str) -> None:
        try:
            self.client.delete_collection(collection_name=collection)
        except Exception:
            pass

    def collection_exists(self, collection: str) -> bool:
        try:
            info = self.client.get_collection(collection_name=collection)
            return info is not None
        except Exception:
            return False

    def list_documents(self, collection: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        results: List[Tuple[str, str, Dict[str, Any]]] = []
        try:
            # use scroll to iterate all points
            for point in self.client.scroll(collection_name=collection, with_payload=True, with_vector=False):
                pid = str(point.id)
                payload = point.payload or {}
                text = payload.get("text") or payload.get("content") or ""
                results.append((pid, text, payload))
        except Exception as exc:
            logger.warning("Failed to list documents from qdrant collection %s: %s", collection, exc)
        return results
