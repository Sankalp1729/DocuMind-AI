from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.security.context import current_tenant_id, current_workspace_id
from backend.rag.multimodal_loader import SUPPORTED_SUFFIXES
from backend.schemas.api import UploadResponse
from backend.services.ingestion_service import IngestionService


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported_types = ", ".join(sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {supported_types}.")

    workspace_service = request.app.state.workspace_service
    workspace_id = request.headers.get("X-Workspace-ID") or request.query_params.get("workspace_id") or request.headers.get("X-Session-ID") or "shared"
    user_id = request.headers.get("X-User-ID")
    workspace_token = current_workspace_id.set(workspace_id)
    tenant_token = current_tenant_id.set(workspace_id)

    try:
        raw_bytes = await file.read()
        workspace, file_path = workspace_service.store_artifact(
            workspace_id=workspace_id,
            filename=file.filename,
            content=raw_bytes,
            artifact_type="document_upload",
            metadata={"content_type": file.content_type},
            user_id=user_id,
        )

        ingestion_service: IngestionService = request.app.state.ingestion_service
        result = ingestion_service.ingest_directory(file_path.parent, workspace_id=workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        current_workspace_id.reset(workspace_token)
        current_tenant_id.reset(tenant_token)

    return UploadResponse(
        message=f"{file.filename} uploaded successfully",
        chunks_created=result.chunks_created,
        stored_file=str(file_path),
        workspace_id=workspace.id,
        files_processed=result.files_processed,
        tables_extracted=result.tables_extracted,
        images_extracted=result.images_extracted,
        ocr_pages=result.ocr_pages,
        average_ocr_confidence=result.average_ocr_confidence,
    )