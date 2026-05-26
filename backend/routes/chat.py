from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.security.context import current_tenant_id, current_workspace_id
from backend.schemas.api import AskResponse, QuestionRequest
from backend.schemas.streaming import StreamingRequest
from backend.services.streaming_service import StreamingService
from backend.utils.token_usage import estimate_turn_tokens


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, payload: QuestionRequest):
    service = request.app.state.rag_service
    conversation_service = request.app.state.conversation_service
    metrics_service = request.app.state.metrics_service
    workspace_service = request.app.state.workspace_service
    session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id") or "anonymous"
    workspace_id = request.headers.get("X-Workspace-ID") or request.query_params.get("workspace_id") or session_id
    workspace_token = current_workspace_id.set(workspace_id)
    tenant_token = current_tenant_id.set(workspace_id)

    try:
        workspace_service.ensure_workspace(workspace_id=workspace_id, user_id=request.headers.get("X-User-ID"))
        conversation_service.create_conversation(session_id, workspace_id=workspace_id)
        conversation_service.append_message(session_id, "user", payload.question, workspace_id=workspace_id)
        result = service.answer(payload.question)
    finally:
        current_workspace_id.reset(workspace_token)
        current_tenant_id.reset(tenant_token)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No vector store is available. Upload a document first or restore vector_store/.",
        )

    conversation_service.append_message(
        session_id,
        "assistant",
        result["answer"],
        sources=result["sources"],
        workspace_id=workspace_id,
    )
    token_usage = estimate_turn_tokens(payload.question, result["answer"], context="\n\n".join(source["preview"] or "" for source in result["sources"]))
    conversation_service.record_token_usage(
        conversation_id=session_id,
        session_id=session_id,
        prompt_tokens=token_usage["prompt_tokens"],
        completion_tokens=token_usage["completion_tokens"],
        total_tokens=token_usage["total_tokens"],
        model_name=None,
        metadata={"mode": "chat"},
    )
    metrics_service.increment("chat_requests_total")

    return AskResponse(**result)


@router.post("/stream")
async def stream_question(request: Request, payload: StreamingRequest):
    rag_service = request.app.state.rag_service
    streaming_service = StreamingService(
        rag_service,
        conversation_service=request.app.state.conversation_service,
        metrics_service=request.app.state.metrics_service,
        cache_service=request.app.state.cache_service,
    )
    session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id") or "anonymous"
    workspace_id = request.headers.get("X-Workspace-ID") or request.query_params.get("workspace_id") or session_id
    workspace_token = current_workspace_id.set(workspace_id)
    tenant_token = current_tenant_id.set(workspace_id)

    try:
        if rag_service.retrieve(payload.question) is None:
            raise HTTPException(
                status_code=404,
                detail="No vector store is available. Upload a document first or restore vector_store/.",
            )
    finally:
        current_workspace_id.reset(workspace_token)
        current_tenant_id.reset(tenant_token)

    return StreamingResponse(
        streaming_service.stream_answer(payload.question, session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )