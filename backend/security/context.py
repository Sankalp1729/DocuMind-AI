from __future__ import annotations

from contextvars import ContextVar


current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
current_workspace_id: ContextVar[str | None] = ContextVar("current_workspace_id", default=None)
current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)
current_roles: ContextVar[tuple[str, ...]] = ContextVar("current_roles", default=())
current_auth_source: ContextVar[str | None] = ContextVar("current_auth_source", default=None)
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)
