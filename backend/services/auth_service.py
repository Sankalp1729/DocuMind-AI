from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import requests
from sqlalchemy import select

from backend.core.config import ADMIN_API_KEY, DATABASE_URL, ENABLE_AUTH
from backend.persistence.auth_models import (
    ApiKeyRecord,
    OAuthIdentityRecord,
    RefreshTokenRecord,
    RoleRecord,
    UserRecord,
    UserRoleRecord,
)
from backend.services.database_service import DatabaseService


logger = logging.getLogger(__name__)

JWT_SECRET_KEY = os.getenv("DOCUMIND_JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = os.getenv("DOCUMIND_JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("DOCUMIND_JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("DOCUMIND_JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
AUTH_PASSWORD_PEPPER = os.getenv("DOCUMIND_AUTH_PASSWORD_PEPPER", "")
GOOGLE_CLIENT_ID = os.getenv("DOCUMIND_GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("DOCUMIND_GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("DOCUMIND_GOOGLE_REDIRECT_URI", "")
AUTH_COOKIE_NAME = os.getenv("DOCUMIND_AUTH_COOKIE_NAME", "documind_session")
AUTH_COOKIE_SECURE = os.getenv("DOCUMIND_AUTH_COOKIE_SECURE", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTH_COOKIE_SAMESITE = os.getenv("DOCUMIND_AUTH_COOKIE_SAMESITE", "lax")

DEFAULT_ROLES = ("owner", "admin", "editor", "viewer", "service")


@dataclass(slots=True)
class AuthContext:
    user: UserRecord | None
    roles: tuple[str, ...]
    auth_source: str | None
    workspace_id: str | None
    tenant_id: str | None
    session_id: str | None
    api_key_id: str | None = None


class AuthService:
    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        with self.database_service.session_scope() as session:
            existing = {role.name for role in session.query(RoleRecord).all()}
            for role_name in DEFAULT_ROLES:
                if role_name not in existing:
                    session.add(RoleRecord(name=role_name, description=f"Default {role_name} role"))

    @staticmethod
    def validate_admin_api_key(provided_key: str | None) -> bool:
        if not provided_key:
            return False
        return hmac.compare_digest(provided_key, ADMIN_API_KEY)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> str:
        salt_value = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            f"{password}{AUTH_PASSWORD_PEPPER}".encode("utf-8"),
            salt_value.encode("utf-8"),
            260000,
        )
        return f"pbkdf2_sha256${salt_value}${base64.urlsafe_b64encode(digest).decode('ascii')}"

    @staticmethod
    def _verify_password(password: str, password_hash: str | None) -> bool:
        if not password_hash:
            return False
        try:
            algorithm, salt, stored_digest = password_hash.split("$", 2)
        except ValueError:
            return False
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = AuthService._hash_password(password, salt=salt)
        return hmac.compare_digest(candidate, password_hash)

    def _role_names_for_user(self, session, user_id: str) -> list[str]:
        rows = (
            session.query(RoleRecord.name)
            .join(UserRoleRecord, UserRoleRecord.role_id == RoleRecord.id)
            .filter(UserRoleRecord.user_id == user_id)
            .all()
        )
        return [row[0] for row in rows]

    def _role_ids(self, session, roles: list[str]) -> list[int]:
        if not roles:
            return []
        rows = session.query(RoleRecord).filter(RoleRecord.name.in_(roles)).all()
        return [row.id for row in rows]

    def _get_or_create_role(self, session, role_name: str) -> RoleRecord:
        role = session.query(RoleRecord).filter(RoleRecord.name == role_name).one_or_none()
        if role is None:
            role = RoleRecord(name=role_name, description=f"{role_name} role")
            session.add(role)
            session.flush()
        return role

    def create_user(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
        tenant_id: str | None = None,
        default_workspace_id: str | None = None,
        roles: list[str] | None = None,
    ) -> UserRecord:
        with self.database_service.session_scope() as session:
            if session.query(UserRecord).filter(UserRecord.email == email.lower()).one_or_none():
                raise ValueError("User already exists")

            user = UserRecord(
                id=uuid.uuid4().hex,
                email=email.lower(),
                full_name=full_name,
                password_hash=self._hash_password(password),
                tenant_id=tenant_id,
                default_workspace_id=default_workspace_id,
                is_active=True,
                is_superuser=False,
                created_at=self._now(),
                updated_at=self._now(),
            )
            session.add(user)
            session.flush()

            assigned_roles = roles or ["viewer"]
            for role_name in assigned_roles:
                role = self._get_or_create_role(session, role_name)
                session.add(UserRoleRecord(user_id=user.id, role_id=role.id))

            session.flush()
            return user

    def authenticate_password(self, email: str, password: str) -> AuthContext | None:
        with self.database_service.session_scope() as session:
            user = session.query(UserRecord).filter(UserRecord.email == email.lower()).one_or_none()
            if user is None or not user.is_active:
                return None
            if not self._verify_password(password, user.password_hash):
                return None
            roles = tuple(self._role_names_for_user(session, user.id))
            return AuthContext(
                user=user,
                roles=roles,
                auth_source="password",
                workspace_id=user.default_workspace_id,
                tenant_id=user.tenant_id,
                session_id=None,
            )

    def _jwt_payload(
        self,
        user: UserRecord,
        roles: tuple[str, ...],
        workspace_id: str | None,
        tenant_id: str | None,
        session_id: str,
        token_type: str,
        expires_delta: timedelta,
        jti: str | None = None,
    ) -> dict[str, Any]:
        issued_at = self._now()
        return {
            "sub": user.id,
            "email": user.email,
            "roles": list(roles),
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "sid": session_id,
            "jti": jti or uuid.uuid4().hex,
            "typ": token_type,
            "iat": int(issued_at.timestamp()),
            "exp": int((issued_at + expires_delta).timestamp()),
        }

    def create_token_pair(
        self,
        user: UserRecord,
        roles: tuple[str, ...] | None = None,
        workspace_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_roles = roles or tuple(self.get_user_roles(user.id))
        resolved_session_id = session_id or uuid.uuid4().hex
        access_payload = self._jwt_payload(
            user=user,
            roles=resolved_roles,
            workspace_id=workspace_id or user.default_workspace_id,
            tenant_id=tenant_id or user.tenant_id,
            session_id=resolved_session_id,
            token_type="access",
            expires_delta=timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        refresh_jti = uuid.uuid4().hex
        refresh_payload = self._jwt_payload(
            user=user,
            roles=resolved_roles,
            workspace_id=workspace_id or user.default_workspace_id,
            tenant_id=tenant_id or user.tenant_id,
            session_id=resolved_session_id,
            token_type="refresh",
            expires_delta=timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            jti=refresh_jti,
        )
        access_token = jwt.encode(access_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        with self.database_service.session_scope() as session:
            session.add(
                RefreshTokenRecord(
                    id=uuid.uuid4().hex,
                    user_id=user.id,
                    session_id=resolved_session_id,
                    tenant_id=tenant_id or user.tenant_id,
                    workspace_id=workspace_id or user.default_workspace_id,
                    jti=refresh_jti,
                    token_hash=self._hash_refresh_token(refresh_token),
                    expires_at=self._now() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS),
                    revoked_at=None,
                    created_at=self._now(),
                    metadata_json={"roles": list(resolved_roles)},
                )
            )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    def _hash_refresh_token(self, token: str) -> str:
        return hashlib.sha256(f"{token}{AUTH_PASSWORD_PEPPER}".encode("utf-8")).hexdigest()

    def verify_refresh_token(self, refresh_token: str) -> AuthContext | None:
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        except jwt.PyJWTError:
            return None

        if payload.get("typ") != "refresh":
            return None

        with self.database_service.session_scope() as session:
            record = session.query(RefreshTokenRecord).filter(RefreshTokenRecord.jti == payload.get("jti")).one_or_none()
            if record is None or record.revoked_at is not None:
                return None
            if record.expires_at <= self._now():
                return None
            user = session.get(UserRecord, record.user_id)
            if user is None or not user.is_active:
                return None
            roles = tuple(self._role_names_for_user(session, user.id))
            return AuthContext(
                user=user,
                roles=roles,
                auth_source="refresh",
                workspace_id=record.workspace_id or user.default_workspace_id,
                tenant_id=record.tenant_id or user.tenant_id,
                session_id=record.session_id,
            )

    def verify_access_token(self, access_token: str) -> AuthContext | None:
        try:
            payload = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        except jwt.PyJWTError:
            return None

        if payload.get("typ") != "access":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        with self.database_service.session_scope() as session:
            user = session.get(UserRecord, user_id)
            if user is None or not user.is_active:
                return None
            roles = tuple(payload.get("roles") or self._role_names_for_user(session, user.id))
            return AuthContext(
                user=user,
                roles=roles,
                auth_source="jwt",
                workspace_id=payload.get("workspace_id") or user.default_workspace_id,
                tenant_id=payload.get("tenant_id") or user.tenant_id,
                session_id=payload.get("sid"),
            )

    def authenticate_api_key(self, api_key: str | None) -> AuthContext | None:
        if not api_key:
            return None
        try:
            prefix, secret = api_key.split(".", 1)
        except ValueError:
            return None

        with self.database_service.session_scope() as session:
            record = session.query(ApiKeyRecord).filter(ApiKeyRecord.prefix == prefix, ApiKeyRecord.revoked_at.is_(None)).one_or_none()
            if record is None:
                return None
            if record.expires_at is not None and record.expires_at <= self._now():
                return None
            expected_hash = record.hashed_secret
            candidate_hash = self._hash_api_secret(secret)
            if not hmac.compare_digest(candidate_hash, expected_hash):
                return None
            record.last_used_at = self._now()
            user = session.get(UserRecord, record.user_id) if record.user_id else None
            roles = tuple(record.roles_json or []) or (tuple(self.get_user_roles(user.id)) if user else ("service",))
            return AuthContext(
                user=user,
                roles=roles,
                auth_source="api_key",
                workspace_id=record.workspace_id or (user.default_workspace_id if user else None),
                tenant_id=user.tenant_id if user else None,
                session_id=None,
                api_key_id=record.id,
            )

    @staticmethod
    def _hash_api_secret(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    def create_api_key(
        self,
        user_id: str | None,
        name: str,
        workspace_id: str | None = None,
        roles: list[str] | None = None,
        expires_in_days: int | None = 30,
    ) -> dict[str, Any]:
        api_key_id = uuid.uuid4().hex
        prefix = secrets.token_hex(4)
        secret = secrets.token_urlsafe(32)
        token_value = f"{prefix}.{secret}"
        expires_at = self._now() + timedelta(days=expires_in_days) if expires_in_days else None

        with self.database_service.session_scope() as session:
            session.add(
                ApiKeyRecord(
                    id=api_key_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    name=name,
                    prefix=prefix,
                    hashed_secret=self._hash_api_secret(secret),
                    roles_json=roles or [],
                    expires_at=expires_at,
                    revoked_at=None,
                    last_used_at=None,
                    created_at=self._now(),
                )
            )

        return {"key_id": api_key_id, "api_key": token_value, "prefix": prefix, "expires_at": expires_at}

    def revoke_api_key(self, key_id: str) -> None:
        with self.database_service.session_scope() as session:
            record = session.get(ApiKeyRecord, key_id)
            if record is not None:
                record.revoked_at = self._now()

    def revoke_refresh_token(self, refresh_jti: str) -> None:
        with self.database_service.session_scope() as session:
            record = session.query(RefreshTokenRecord).filter(RefreshTokenRecord.jti == refresh_jti).one_or_none()
            if record is not None:
                record.revoked_at = self._now()

    def get_user_roles(self, user_id: str) -> list[str]:
        with self.database_service.session_scope() as session:
            return self._role_names_for_user(session, user_id)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with self.database_service.session_scope() as session:
            return session.query(UserRecord).filter(UserRecord.email == email.lower()).one_or_none()

    def get_user_by_id(self, user_id: str) -> UserRecord | None:
        with self.database_service.session_scope() as session:
            return session.get(UserRecord, user_id)

    def ensure_user_from_google_profile(
        self,
        provider_subject: str,
        email: str,
        full_name: str | None,
        tenant_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserRecord:
        with self.database_service.session_scope() as session:
            identity = (
                session.query(OAuthIdentityRecord)
                .filter(OAuthIdentityRecord.provider == "google", OAuthIdentityRecord.provider_subject == provider_subject)
                .one_or_none()
            )
            if identity is not None:
                user = session.get(UserRecord, identity.user_id)
                if user is not None:
                    user.email = email.lower()
                    user.full_name = full_name or user.full_name
                    user.updated_at = self._now()
                    return user

            user = session.query(UserRecord).filter(UserRecord.email == email.lower()).one_or_none()
            if user is None:
                user = UserRecord(
                    id=uuid.uuid4().hex,
                    email=email.lower(),
                    full_name=full_name,
                    password_hash=None,
                    tenant_id=tenant_id,
                    default_workspace_id=workspace_id,
                    is_active=True,
                    is_superuser=False,
                    created_at=self._now(),
                    updated_at=self._now(),
                )
                session.add(user)
                session.flush()
                role = self._get_or_create_role(session, "viewer")
                session.add(UserRoleRecord(user_id=user.id, role_id=role.id))
            session.add(
                OAuthIdentityRecord(
                    user_id=user.id,
                    provider="google",
                    provider_subject=provider_subject,
                    email=email.lower(),
                    metadata_json=metadata or {},
                    created_at=self._now(),
                )
            )
            return user

    def google_login(
        self,
        *,
        code: str | None = None,
        id_token: str | None = None,
        access_token: str | None = None,
        redirect_uri: str | None = None,
        workspace_id: str | None = None,
        tenant_id: str | None = None,
    ) -> tuple[AuthContext, dict[str, Any]]:
        profile = self._google_profile(code=code, id_token=id_token, access_token=access_token, redirect_uri=redirect_uri)
        user = self.ensure_user_from_google_profile(
            provider_subject=profile["sub"],
            email=profile["email"],
            full_name=profile.get("name"),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=profile,
        )
        roles = tuple(self.get_user_roles(user.id))
        auth_context = AuthContext(
            user=user,
            roles=roles,
            auth_source="google",
            workspace_id=workspace_id or user.default_workspace_id,
            tenant_id=tenant_id or user.tenant_id,
            session_id=None,
        )
        return auth_context, self.create_token_pair(user, roles=roles, workspace_id=auth_context.workspace_id, tenant_id=auth_context.tenant_id)

    def _google_profile(
        self,
        *,
        code: str | None,
        id_token: str | None,
        access_token: str | None,
        redirect_uri: str | None,
    ) -> dict[str, Any]:
        if id_token:
            response = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if GOOGLE_CLIENT_ID and payload.get("aud") not in {GOOGLE_CLIENT_ID}:
                raise ValueError("Google token audience mismatch")
            return payload

        if access_token:
            response = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
            response.raise_for_status()
            return response.json()

        if code:
            if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
                raise ValueError("Google OAuth client is not configured")
            token_response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": redirect_uri or GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                timeout=10,
            )
            token_response.raise_for_status()
            token_payload = token_response.json()
            if token_payload.get("id_token"):
                return self._google_profile(code=None, id_token=token_payload["id_token"], access_token=None, redirect_uri=None)
            if token_payload.get("access_token"):
                return self._google_profile(code=None, id_token=None, access_token=token_payload["access_token"], redirect_uri=None)

        raise ValueError("Google OAuth credentials were not provided")

    def introspect_auth_header(self, authorization: str | None, api_key: str | None) -> AuthContext | None:
        if api_key:
            api_context = self.authenticate_api_key(api_key)
            if api_context is not None:
                return api_context

        if not authorization:
            return None

        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return self.verify_access_token(token)

        if "." in authorization:
            return self.verify_access_token(authorization)

        return None

    def require_roles(self, context: AuthContext | None, allowed_roles: set[str]) -> bool:
        if context is None:
            return False
        if not ENABLE_AUTH:
            return True
        if "owner" in context.roles or "admin" in context.roles:
            return True
        return any(role in allowed_roles for role in context.roles)


def validate_admin_api_key(provided_key: str | None) -> bool:
    if not provided_key:
        return False
    return hmac.compare_digest(provided_key, ADMIN_API_KEY)