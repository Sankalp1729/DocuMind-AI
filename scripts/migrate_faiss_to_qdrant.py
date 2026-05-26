"""Simple migration script to copy FAISS local indexes into Qdrant collections.

Usage:
    python scripts/migrate_faiss_to_qdrant.py /path/to/vector_store base_collection_prefix

This will iterate subdirectories under the FAISS persist directory, load each index,
extract documents and embeddings (if stored in metadata), and upsert them to a Qdrant
collection named {base_collection_prefix}_{namespace}.

Note: Embeddings must be available in the FAISS docstore payload metadata under
"embedding". If not present, the script will attempt to compute embeddings which
requires the embedding model and memory.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List

from backend.rag.vector_store import load_vector_store, get_embeddings
from backend.services.qdrant_backend import QdrantBackend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def migrate(faiss_root: Path, qdrant_url: str = "localhost", qdrant_port: int = 6333, prefix: str = "documind"):
    q = QdrantBackend(url=qdrant_url, port=qdrant_port)
    for child in faiss_root.iterdir():
        if not child.is_dir():
            continue
        namespace = child.name
        logger.info("Loading FAISS index from %s", child)
        vs = load_vector_store(child)
        if vs is None:
            logger.warning("No index found in %s, skipping", child)
            continue

        # extract docs from FAISS docstore
        docs = []
        try:
            docstore = getattr(vs, "docstore", None)
            index_to_docstore_id = getattr(vs, "index_to_docstore_id", {})
            for idx in range(len(index_to_docstore_id)):
                doc_id = index_to_docstore_id[idx]
                doc = docstore.search(doc_id)
                if doc:
                    docs.append((str(doc_id), doc.page_content, doc.metadata or {}))
        except Exception as exc:
            logger.warning("Failed to iterate FAISS docstore for %s: %s", child, exc)
            continue

        if not docs:
            logger.warning("No documents to migrate for %s", child)
            continue

        # prepare vectors
        embedder = get_embeddings()
        points: List[tuple] = []
        for did, text, metadata in docs:
            vec = metadata.get("embedding")
            if vec is None:
                vec = embedder.embed_query(text)
            payload = {**(metadata or {}), "text": text}
            points.append((did, vec, payload))

        collection = f"{prefix}_{namespace}"
        # ensure collection exists
        q.create_collection(collection, vector_size=len(points[0][1]))
        logger.info("Upserting %s points into Qdrant collection %s", len(points), collection)
        q.upsert(collection, points)
        logger.info("Migration for %s completed", namespace)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/migrate_faiss_to_qdrant.py /path/to/faiss_root qdrant_prefix [qdrant_url] [qdrant_port]")
        sys.exit(1)

    root = Path(sys.argv[1])
    prefix = sys.argv[2]
    url = sys.argv[3] if len(sys.argv) > 3 else "localhost"
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 6333
    migrate(root, qdrant_url=url, qdrant_port=port, prefix=prefix)
