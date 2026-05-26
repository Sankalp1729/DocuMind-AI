from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

from backend.core import config
from backend.core.config import VECTOR_BACKEND
from backend.security.context import current_workspace_id, current_tenant_id
from backend.services.vector_db import VectorDB
from backend.services.qdrant_backend import QdrantBackend
from backend.rag.vector_store import create_vector_store, load_vector_store, save_vector_store


logger = logging.getLogger(__name__)


class VectorStoreService:
    """Unified service that supports either FAISS filesystem or a remote vector DB like Qdrant.

    - For `faiss` backend we persist local FAISS indexes per-namespace.
    - For `qdrant` backend we manage collections per-namespace.
    """

    def __init__(self, persist_directory=Path(config.VECTOR_STORE_DIR)):
        self.persist_directory = Path(persist_directory)
        self._lock = RLock()
        self._vector_stores: dict[str, Any] = {}
        self.backend: VectorDB | None = None
        self.backend_type = VECTOR_BACKEND.lower()
        if self.backend_type == "qdrant":
            self.backend = QdrantBackend(url=config.QDRANT_URL, port=config.QDRANT_PORT, prefer_grpc=config.QDRANT_PREFER_GRPC)

    def _namespace(self, tenant_id: str | None = None) -> str:
        namespace = tenant_id or current_workspace_id.get() or current_tenant_id.get() or "shared"
        return namespace.replace("/", "_")

    def _persist_directory(self, tenant_id: str | None = None) -> Path:
        return self.persist_directory / self._namespace(tenant_id)

    # FAISS-based operations (local) - kept for migration/backwards compat
    def _load_faiss(self, tenant_id: str | None = None):
        persist_directory = self._persist_directory(tenant_id)
        return load_vector_store(persist_directory)

    def _save_faiss(self, vector_store=None, tenant_id: str | None = None):
        persist_directory = self._persist_directory(tenant_id)
        return save_vector_store(vector_store, persist_directory)

    def load_vector_store(self, tenant_id: str | None = None):
        namespace = self._namespace(tenant_id)
        with self._lock:
            if namespace in self._vector_stores:
                return self._vector_stores[namespace]

            if self.backend_type == "qdrant" and self.backend is not None:
                # Qdrant doesn't return a local object; return backend and collection name
                if self.backend.collection_exists(namespace):
                    self._vector_stores[namespace] = (self.backend, namespace)
                    return self._vector_stores[namespace]
                return None

            vector_store = self._load_faiss(tenant_id)
            if vector_store is None:
                logger.info("No persisted vector store found in %s", self._persist_directory(namespace))
                return None

            self._vector_stores[namespace] = vector_store
            return self._vector_stores[namespace]

    def save_vector_store(self, vector_store=None, tenant_id: str | None = None):
        namespace = self._namespace(tenant_id)
        with self._lock:
            if self.backend_type == "qdrant" and self.backend is not None:
                # Qdrant persists remotely; no-op
                if not self.backend.collection_exists(namespace):
                    # assume vector size must be inferred by caller; skip creation here
                    pass
                self._vector_stores[namespace] = (self.backend, namespace)
                return self._vector_stores[namespace]

            active_vector_store = vector_store or self._vector_stores.get(namespace)
            if active_vector_store is None:
                raise ValueError("No vector store is available to persist")

            self._vector_stores[namespace] = self._save_faiss(active_vector_store, tenant_id)
            return self._vector_stores[namespace]

    def rebuild_vector_store(self, chunks, tenant_id: str | None = None):
        namespace = self._namespace(tenant_id)
        with self._lock:
            if not chunks:
                raise ValueError("Cannot rebuild vector store from an empty chunk list")

            if self.backend_type == "qdrant" and self.backend is not None:
                # Upsert all chunk vectors into qdrant collection
                vectors = []
                for chunk in chunks:
                    vec = chunk.metadata.get("embedding") if chunk.metadata else None
                    if vec is None:
                        # compute embedding on the fly using existing helper
                        emb = None
                    else:
                        emb = vec
                    payload = {**(chunk.metadata or {}), "text": chunk.page_content}
                    vectors.append((chunk.metadata.get("chunk_id") or chunk.metadata.get("source_file") or chunk.id, emb, payload))
                # create collection if necessary
                if not self.backend.collection_exists(namespace):
                    # infer vector size from first vector
                    vector_size = len(vectors[0][1]) if vectors and vectors[0][1] is not None else 1536
                    self.backend.create_collection(namespace, vector_size=vector_size)
                self.backend.upsert(namespace, vectors)
                self._vector_stores[namespace] = (self.backend, namespace)
                return self._vector_stores[namespace]

            # FAISS rebuild
            self._vector_stores[namespace] = create_vector_store(chunks, persist_directory=self._persist_directory(namespace))
            return self._vector_stores[namespace]

    def get_vector_store(self, tenant_id: str | None = None):
        namespace = self._namespace(tenant_id)
        with self._lock:
            if namespace in self._vector_stores:
                return self._vector_stores[namespace]

        return self.load_vector_store(tenant_id=tenant_id)

    def is_ready(self, tenant_id: str | None = None):
        if self.backend_type == "qdrant" and self.backend is not None:
            return self.backend.collection_exists(self._namespace(tenant_id))
        return self.get_vector_store(tenant_id=tenant_id) is not None