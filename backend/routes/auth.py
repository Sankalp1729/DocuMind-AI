from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from backend.core.config import AUTH_COOKIE_NAME, AUTH_COOKIE_REFRESH_NAME, AUTH_COOKIE_SAMESITE, AUTH_COOKIE_SECURE
from backend.schemas.auth import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    AuthMeResponse,
    AuthenticatedUserResponse,
    LoginRequest,
    OAuthLoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPairResponse,
)


router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user, roles: list[str], workspace_id: str | None, tenant_id: str | None) -> AuthenticatedUserResponse:
    return AuthenticatedUserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        roles=roles,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        is_active=user.is_active,
    )


def _set_auth_cookies(response: Response, tokens: dict[str, str]) -> None:
    response.set_cookie(
        AUTH_COOKIE_NAME,
        tokens["access_token"],
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        max_age=tokens["expires_in"],
        path="/",
    )
    response.set_cookie(
        AUTH_COOKIE_REFRESH_NAME,
        tokens["refresh_token"],
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


@router.post("/register", response_model=TokenPairResponse)
def register_user(request: Request, payload: RegisterRequest):
    auth_service = request.app.state.auth_service
    try:
        user = auth_service.create_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            workspace_id=payload.workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    tokens = auth_service.create_token_pair(user, workspace_id=payload.workspace_id or user.default_workspace_id, tenant_id=user.tenant_id)
    response = JSONResponse(tokens)
    _set_auth_cookies(response, tokens)
    return response


@router.post("/login", response_model=TokenPairResponse)
def login_user(request: Request, payload: LoginRequest):
    auth_service = request.app.state.auth_service
    context = auth_service.authenticate_password(payload.email, payload.password)
    if context is None or context.user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    tokens = auth_service.create_token_pair(
        context.user,
        roles=context.roles,
        workspace_id=payload.workspace_id or context.workspace_id,
        tenant_id=context.tenant_id,
    )
    response = JSONResponse(tokens)
    _set_auth_cookies(response, tokens)
    return response


@router.post("/oauth/google", response_model=TokenPairResponse)
def google_login(request: Request, payload: OAuthLoginRequest):
    auth_service = request.app.state.auth_service
    try:
        _, tokens = auth_service.google_login(
            code=payload.code,
            id_token=payload.id_token,
            access_token=payload.access_token,
            redirect_uri=payload.redirect_uri,
            workspace_id=payload.workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = JSONResponse(tokens)
    _set_auth_cookies(response, tokens)
    return response


@router.post("/refresh", response_model=TokenPairResponse)
def refresh_tokens(request: Request, payload: RefreshRequest):
    auth_service = request.app.state.auth_service
    context = auth_service.verify_refresh_token(payload.refresh_token)
    if context is None or context.user is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    auth_service.revoke_refresh_token(payload.refresh_token)
    tokens = auth_service.create_token_pair(
        context.user,
        roles=context.roles,
        workspace_id=context.workspace_id,
        tenant_id=context.tenant_id,
        session_id=context.session_id,
    )
    response = JSONResponse(tokens)
    _set_auth_cookies(response, tokens)
    return response


@router.get("/me", response_model=AuthMeResponse)
def auth_me(request: Request):
    user = getattr(request.state, "current_user", None)
    roles = list(getattr(request.state, "current_roles", ()) or [])
    user_response = _user_response(user, roles, getattr(request.state, "current_workspace_id", None), getattr(request.state, "current_tenant_id", None)) if user else None
    return AuthMeResponse(
        user=user_response,
        auth_source=getattr(request.state, "auth_source", None),
        workspace_id=getattr(request.state, "current_workspace_id", None),
        tenant_id=getattr(request.state, "current_tenant_id", None),
        session_id=getattr(request.state, "current_session_id", None),
        api_key_id=getattr(request.state, "api_key_id", None),
        permissions=roles,
        metadata={"authenticated": bool(user)},
    )


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
def create_api_key(request: Request, payload: ApiKeyCreateRequest):
    auth_service = request.app.state.auth_service
    current_user = getattr(request.state, "current_user", None)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    created = auth_service.create_api_key(
        user_id=current_user.id,
        name=payload.name,
        workspace_id=payload.workspace_id or getattr(request.state, "current_workspace_id", None),
        roles=[role for role in payload.roles],
        expires_in_days=payload.expires_in_days,
    )
    return ApiKeyCreateResponse(**created)
