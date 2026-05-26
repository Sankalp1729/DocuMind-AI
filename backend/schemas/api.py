from pydantic import BaseModel, Field
from typing import Optional

from backend.schemas.retrieval import RetrievalExplanation, GroundednessScore


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceCitation(BaseModel):
    source: str | None = None
    page: int | None = None
    preview: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]
    retrieval_explanation: Optional[RetrievalExplanation] = None
    groundedness: Optional[GroundednessScore] = None
    agent_plan: Optional[dict] = None
    agent_reflections: list[str] | None = None
    agent_observations: list[dict] | None = None


class UploadResponse(BaseModel):
    message: str
    chunks_created: int
    stored_file: str
    files_processed: int | None = None
    tables_extracted: int | None = None
    images_extracted: int | None = None
    ocr_pages: int | None = None
    average_ocr_confidence: float | None = None


class IngestionResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    persist_directory: str
    files_processed: int | None = None
    tables_extracted: int | None = None
    images_extracted: int | None = None
    ocr_pages: int | None = None
    average_ocr_confidence: float | None = None


class HealthResponse(BaseModel):
    message: str
    vector_store_ready: bool