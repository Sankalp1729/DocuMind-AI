from __future__ import annotations

import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.core.config import AUTH_COOKIE_NAME, AUTH_PUBLIC_PATHS, ENABLE_AUTH, RATE_LIMIT_PER_MINUTE, TENANT_HEADER_NAME, USER_HEADER_NAME, API_KEY_HEADER_NAME
from backend.core.logging import request_id_context
from backend.security.context import (
    current_auth_source,
    current_roles,
    current_session_id,
    current_tenant_id,
    current_user_id,
    current_workspace_id,
)


def _is_public_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in AUTH_PUBLIC_PATHS)


class AuthContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tokens = []
        auth_service = getattr(request.app.state, "auth_service", None)
        authorization = request.headers.get("Authorization")
        api_key = request.headers.get(API_KEY_HEADER_NAME)
        session_token = request.cookies.get(AUTH_COOKIE_NAME)

        auth_context = None
        if auth_service is not None:
            auth_context = auth_service.introspect_auth_header(authorization or session_token, api_key)

        if auth_context is None:
            user_id = request.headers.get(USER_HEADER_NAME)
            workspace_id = request.headers.get(TENANT_HEADER_NAME)
            request.state.authenticated = False
            request.state.current_user = None
            request.state.current_roles = ()
            request.state.auth_source = None
            request.state.current_user_id = user_id
            request.state.current_workspace_id = workspace_id
            request.state.current_tenant_id = workspace_id
            request.state.current_session_id = request.headers.get("X-Session-ID")
            if ENABLE_AUTH and not _is_public_path(request.url.path):
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
        else:
            request.state.authenticated = True
            request.state.current_user = auth_context.user
            request.state.current_roles = auth_context.roles
            request.state.auth_source = auth_context.auth_source
            request.state.current_user_id = auth_context.user.id if auth_context.user else None
            request.state.current_workspace_id = auth_context.workspace_id or request.headers.get(TENANT_HEADER_NAME)
            request.state.current_tenant_id = auth_context.tenant_id or request.state.current_workspace_id
            request.state.current_session_id = auth_context.session_id or request.headers.get("X-Session-ID")

            # Enforce tenant/workspace header binding: if a client supplied a workspace header
            # it must match the workspace in the token unless the user is admin/owner or using an api key
            header_workspace = request.headers.get(TENANT_HEADER_NAME)
            if header_workspace and auth_context is not None:
                roles = set(auth_context.roles or ())
                if header_workspace != (auth_context.workspace_id or auth_context.tenant_id):
                    if not ("admin" in roles or "owner" in roles or auth_context.auth_source == "api_key"):
                        return JSONResponse(status_code=403, content={"detail": "Workspace header does not match authenticated workspace"})

        token_user_id = current_user_id.set(request.state.current_user_id)
        token_workspace_id = current_workspace_id.set(request.state.current_workspace_id)
        token_tenant_id = current_tenant_id.set(request.state.current_tenant_id)
        token_roles = current_roles.set(tuple(request.state.current_roles or ()))
        token_auth_source = current_auth_source.set(request.state.auth_source)
        token_session_id = current_session_id.set(request.state.current_session_id)
        tokens.extend([token_user_id, token_workspace_id, token_tenant_id, token_roles, token_auth_source, token_session_id])

        try:
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            return response
        finally:
            for token in reversed(tokens):
                try:
                    if token is token_user_id:
                        current_user_id.reset(token)
                    elif token is token_workspace_id:
                        current_workspace_id.reset(token)
                    elif token is token_tenant_id:
                        current_tenant_id.reset(token)
                    elif token is token_roles:
                        current_roles.reset(token)
                    elif token is token_auth_source:
                        current_auth_source.reset(token)
                    elif token is token_session_id:
                        current_session_id.reset(token)
                except Exception:
                    continue


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        token = request_id_context.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_context.reset(token)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = RATE_LIMIT_PER_MINUTE):
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(("/health", "/admin")):
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._requests[client_host]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self.limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
            )

        window.append(now)
        return await call_next(request)