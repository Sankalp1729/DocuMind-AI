import logging
from dataclasses import dataclass
from pathlib import Path

from backend.core.config import DATA_DIR
from backend.rag.multimodal_loader import load_multimodal_documents
from backend.rag.text_splitter import split_documents
from backend.services.vector_store_service import VectorStoreService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    documents_loaded: int
    chunks_created: int
    persist_directory: str
    files_processed: int = 0
    tables_extracted: int = 0
    images_extracted: int = 0
    ocr_pages: int = 0
    average_ocr_confidence: float | None = None


class IngestionService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    def ingest_directory(self, data_dir: Path | None = None, workspace_id: str | None = None) -> IngestionResult:
        target_dir = data_dir or DATA_DIR

        loaded = load_multimodal_documents(str(target_dir))
        documents = loaded.documents
        if not documents:
            raise ValueError(f"No documents found in {target_dir}")

        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were generated from the loaded documents")

        self.vector_store_service.rebuild_vector_store(chunks, workspace_id=workspace_id)

        logger.info(
            "Ingestion complete: %s documents -> %s chunks into %s",
            len(documents),
            len(chunks),
            str(self.vector_store_service._workspace_path(workspace_id or None)),
        )

        return IngestionResult(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            persist_directory=str(self.vector_store_service._workspace_path(workspace_id or None)),
            files_processed=loaded.stats.files_processed,
            tables_extracted=loaded.stats.tables_extracted,
            images_extracted=loaded.stats.images_extracted,
            ocr_pages=loaded.stats.ocr_pages,
            average_ocr_confidence=loaded.stats.average_ocr_confidence,
        )