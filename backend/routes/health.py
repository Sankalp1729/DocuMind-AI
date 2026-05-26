from fastapi import APIRouter, Request

from backend.schemas.api import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    service = request.app.state.vector_store_service
    return HealthResponse(
        message="DocuMind AI backend is healthy",
        vector_store_ready=service.is_ready(),
    )