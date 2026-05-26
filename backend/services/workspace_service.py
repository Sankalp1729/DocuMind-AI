from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.core.config import WORKSPACE_DIR
from backend.persistence.models import WorkspaceArtifactRecord, WorkspaceRecord
from backend.services.database_service import DatabaseService


class WorkspaceService:
    def __init__(self, database_service: DatabaseService, workspace_root: Path | None = None):
        self.database_service = database_service
        self.workspace_root = Path(workspace_root or WORKSPACE_DIR)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _workspace_path(self, workspace_id: str) -> Path:
        path = self.workspace_root / workspace_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_workspace(self, workspace_id: str | None = None, user_id: str | None = None, name: str | None = None) -> WorkspaceRecord:
        resolved_workspace_id = workspace_id or user_id or "shared"
        now = datetime.now(timezone.utc)
        storage_path = str(self._workspace_path(resolved_workspace_id))

        with self.database_service.session_scope() as session:
            workspace = session.get(WorkspaceRecord, resolved_workspace_id)
            if workspace is None:
                workspace = WorkspaceRecord(
                    id=resolved_workspace_id,
                    user_id=user_id,
                    name=name or resolved_workspace_id,
                    storage_path=storage_path,
                    created_at=now,
                    updated_at=now,
                )
                session.add(workspace)
            else:
                workspace.user_id = user_id or workspace.user_id
                workspace.name = name or workspace.name
                workspace.storage_path = storage_path
                workspace.updated_at = now
            session.flush()
            return workspace

    def list_workspaces(self, user_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.database_service.session_scope() as session:
            query = session.query(WorkspaceRecord).order_by(WorkspaceRecord.updated_at.desc())
            if user_id:
                query = query.filter(WorkspaceRecord.user_id == user_id)
            rows = query.limit(limit).all()

        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "name": row.name,
                "storage_path": row.storage_path,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ]

    def store_artifact(
        self,
        workspace_id: str | None,
        filename: str,
        content: bytes,
        artifact_type: str = "upload",
        metadata: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> tuple[WorkspaceRecord, Path]:
        workspace = self.ensure_workspace(workspace_id=workspace_id, user_id=user_id)
        workspace_path = self._workspace_path(workspace.id)
        stored_name = f"{uuid4().hex}_{Path(filename).name}"
        stored_path = workspace_path / stored_name
        stored_path.write_bytes(content)

        now = datetime.now(timezone.utc)
        with self.database_service.session_scope() as session:
            session.add(
                WorkspaceArtifactRecord(
                    workspace_id=workspace.id,
                    artifact_type=artifact_type,
                    file_name=Path(filename).name,
                    file_path=str(stored_path),
                    metadata_json=metadata or {},
                    created_at=now,
                )
            )
            workspace.updated_at = now
            session.merge(workspace)

        return workspace, stored_path
