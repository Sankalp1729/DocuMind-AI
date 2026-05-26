from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/{conversation_id}")
def get_conversation_history(conversation_id: str, request: Request):
    service = request.app.state.conversation_service
    return {"conversation_id": conversation_id, "messages": service.get_history(conversation_id)}


@router.get("")
def list_conversations(request: Request, limit: int = 20):
    service = request.app.state.conversation_service
    return {"items": service.list_conversations(limit=limit)}