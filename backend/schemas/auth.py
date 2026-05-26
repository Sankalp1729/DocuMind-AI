from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


RoleName = Literal["owner", "admin", "editor", "viewer", "service"]


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    full_name: str | None = Field(default=None, max_length=256)
    workspace_id: str | None = Field(default=None, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    workspace_id: str | None = Field(default=None, max_length=128)


class OAuthLoginRequest(BaseModel):
    provider: Literal["google"] = "google"
    code: str | None = None
    id_token: str | None = None
    access_token: str | None = None
    redirect_uri: str | None = None
    workspace_id: str | None = Field(default=None, max_length=128)


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthenticatedUserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None = None
    roles: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    workspace_id: str | None = None
    is_active: bool = True


class RefreshRequest(BaseModel):
    refresh_token: str


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    workspace_id: str | None = Field(default=None, max_length=128)
    roles: list[RoleName] = Field(default_factory=list)
    expires_in_days: int | None = Field(default=30, ge=1, le=3650)


class ApiKeyCreateResponse(BaseModel):
    key_id: str
    api_key: str
    prefix: str
    expires_at: datetime | None = None


class AuthMeResponse(BaseModel):
    user: AuthenticatedUserResponse | None = None
    auth_source: str | None = None
    workspace_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    api_key_id: str | None = None
    permissions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
