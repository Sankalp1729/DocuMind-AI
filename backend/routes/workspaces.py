from __future__ import annotations

from fastapi import APIRouter, Header, Request


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("")
def list_workspaces(request: Request, limit: int = 100, x_user_id: str | None = Header(default=None)):
    service = request.app.state.workspace_service
    return {"items": service.list_workspaces(user_id=x_user_id, limit=limit)}


@router.post("")
def create_workspace(request: Request, workspace_id: str | None = None, name: str | None = None, x_user_id: str | None = Header(default=None)):
    service = request.app.state.workspace_service
    workspace = service.ensure_workspace(workspace_id=workspace_id, user_id=x_user_id, name=name)
    return {
        "id": workspace.id,
        "user_id": workspace.user_id,
        "name": workspace.name,
        "storage_path": workspace.storage_path,
        "created_at": workspace.created_at.isoformat(),
        "updated_at": workspace.updated_at.isoformat(),
    }
