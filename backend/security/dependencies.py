from __future__ import annotations

from typing import Callable, Iterable, Set

from fastapi import Depends, HTTPException, status

from backend.services.auth_service import AuthService
from backend.security import context as security_context


def require_roles_factory(auth_service: AuthService, allowed_roles: Iterable[str]) -> Callable:
    allowed_set: Set[str] = set(allowed_roles)

    def _dependency() -> None:
        ctx_roles = set(security_context.current_roles.get() or ())
        if not auth_service.require_roles(None, allowed_set) and not (ctx_roles & allowed_set):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

    return _dependency
