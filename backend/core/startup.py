import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.logging import configure_logging
from backend.services.cache_service import CacheService
from backend.services.auth_service import AuthService
from backend.services.conversation_service import ConversationService
from backend.services.database_service import DatabaseService
from backend.services.ingestion_service import IngestionService
from backend.services.metrics_service import MetricsService
from backend.services.rag_service import RagService
from backend.services.workspace_service import WorkspaceService
from backend.services.vector_store_service import VectorStoreService
from backend.services.retrieval_service import RetrievalService
from backend.services.telemetry_service import TelemetryService
from backend.core.tracing import init_tracing
from backend.core.metrics_middleware import PrometheusMiddleware, metrics_endpoint
from backend.core.metrics import REQUEST_COUNT
from backend.services.evaluation_service import EvaluationService
from backend.agentic.service import AgenticRagService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    database_service = DatabaseService()
    cache_service = CacheService()
    auth_service = AuthService(database_service=database_service)
    vector_store_service = VectorStoreService()
    workspace_service = WorkspaceService(database_service=database_service)
    conversation_service = ConversationService(database_service=database_service, cache_service=cache_service, workspace_service=workspace_service)
    metrics_service = MetricsService(database_service=database_service, cache_service=cache_service)
    telemetry_service = TelemetryService(database_service=database_service)
    evaluation_service = EvaluationService()
    retrieval_service = RetrievalService(vector_store_service=vector_store_service, telemetry_service=telemetry_service)
    rag_service = RagService(
        vector_store_service=vector_store_service,
        retrieval_service=retrieval_service,
        telemetry_service=telemetry_service,
        cache_service=cache_service,
    )
    agentic_service = AgenticRagService(rag_service=rag_service)
    rag_service.agentic_service = agentic_service
    ingestion_service = IngestionService(vector_store_service=vector_store_service)

    app.state.database_service = database_service
    app.state.cache_service = cache_service
    app.state.auth_service = auth_service
    app.state.vector_store_service = vector_store_service
    app.state.conversation_service = conversation_service
    app.state.metrics_service = metrics_service
    app.state.telemetry_service = telemetry_service
    app.state.evaluation_service = evaluation_service
    app.state.retrieval_service = retrieval_service
    app.state.rag_service = rag_service
    app.state.agentic_rag_service = agentic_service
    app.state.ingestion_service = ingestion_service
    app.state.workspace_service = workspace_service

    # Initialize tracing and metrics
    init_tracing(app)

    vector_store = vector_store_service.load_vector_store()
    if vector_store is None:
        logger.info("No persisted FAISS index found at %s", vector_store_service.persist_directory)
    else:
        logger.info("Loaded persisted FAISS index from %s", vector_store_service.persist_directory)

    try:
        yield
    finally:
        logger.info("Shutting down DocuMind AI backend")