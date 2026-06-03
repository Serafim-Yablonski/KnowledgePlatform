"""MCP authentication: pure ASGI middleware + per-request ContextVar."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import secrets
import uuid
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.core.config import settings
from src.core.database import get_session
from src.core.exceptions import AppError, ForbiddenError, UnauthorizedError
from src.services.auth import AuthService

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from src.models.user import User

# Carries the authenticated User for the duration of a single HTTP request.
_current_user: contextvars.ContextVar[User | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)

_MCP_SESSION_HEADER = "mcp-session-id"


async def _update_last_used(key_id: uuid.UUID) -> None:
    """Fire-and-forget coroutine: update last_used_at in a fresh session."""
    from datetime import UTC, datetime

    import sqlalchemy as sa

    from src.models.api_key import ApiKey

    try:
        async with get_session() as session:
            await session.execute(
                sa.update(ApiKey)
                .where(ApiKey.id == key_id)
                .values(last_used_at=datetime.now(UTC))
            )
            await session.commit()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("failed to update api key last_used_at", key_id=str(key_id))


async def _authenticate_jwt(token: str) -> User:
    async with get_session() as session:
        from src.repositories.user import SQLAlchemyUserRepository

        repo = SQLAlchemyUserRepository(session)
        service = AuthService(repo)
        return await service.get_current_user(token)


async def _authenticate_db_api_key(raw_key: str, session: AsyncSession) -> User:
    """Authenticate against a database-stored API key (hash lookup)."""
    from src.core.cache import ResponseCache
    from src.core.redis import get_redis
    from src.repositories.api_key import SQLAlchemyApiKeyRepository
    from src.repositories.api_key_cached import CachedApiKeyRepository

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    cache = ResponseCache(get_redis())
    repo = CachedApiKeyRepository(SQLAlchemyApiKeyRepository(session), cache)
    api_key = await repo.get_by_hash(key_hash)
    if not api_key or not api_key.is_active:
        raise ForbiddenError("Invalid or revoked API key")
    if not api_key.user.is_active:
        raise ForbiddenError("Invalid or revoked API key")
    asyncio.create_task(_update_last_used(api_key.id))
    return api_key.user


def _looks_like_jwt(token: str) -> bool:
    """Return True if the token is structurally a JWT (header.payload.signature)."""
    return token.count(".") == 2


async def _authenticate_request(request: Request) -> User:
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

        if _looks_like_jwt(token):
            # Definitely a JWT — always use the JWT path and surface the real
            # error (expired, bad signature, etc.) instead of masking it with
            # "Invalid or revoked API key".
            return await _authenticate_jwt(token)

        async with get_session() as session:
            return await _authenticate_db_api_key(token, session)

    # Legacy static API key (env-var pair) — kept for backward compatibility.
    api_key_header = request.headers.get("X-API-Key", "")
    configured_key = settings.MCP_API_KEY
    if (
        api_key_header
        and configured_key
        and secrets.compare_digest(api_key_header, configured_key)
    ):
        if not settings.MCP_API_KEY_USER_EMAIL:
            raise ForbiddenError("API key user email not configured")
        async with get_session() as session:
            from src.repositories.user import SQLAlchemyUserRepository

            repo = SQLAlchemyUserRepository(session)
            user = await repo.get_by_email(settings.MCP_API_KEY_USER_EMAIL)
            if user is None or not user.is_active:
                raise ForbiddenError("API key user not found or inactive")
            return user

    raise UnauthorizedError(
        "Authentication required — provide Authorization: Bearer <token> "
        "or X-API-Key header"
    )


def get_mcp_user() -> User:
    """Return the authenticated user for the current MCP request.

    Raises UnauthorizedError if called outside an authenticated request context.
    """
    user = _current_user.get()
    if user is None:
        raise UnauthorizedError("Authentication required")
    return user


class MCPAuthMiddleware:
    """Pure ASGI middleware that validates auth before FastMCP sees the request.

    Uses a pure ASGI approach (not BaseHTTPMiddleware) to avoid buffering
    streaming SSE responses produced by the Streamable HTTP transport.
    Only http scopes are authenticated; lifespan passes through untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return

        if scope["type"] != "http":
            response = JSONResponse(
                {"error": {"code": 403, "message": "Forbidden"}}, status_code=403
            )
            await response(scope, receive, send)
            return

        request = Request(scope)
        try:
            user = await _authenticate_request(request)
        except AppError as exc:
            response = JSONResponse(
                {"error": {"code": exc.status_code, "message": exc.detail}},
                status_code=exc.status_code,
            )
            await response(scope, receive, send)
            return
        except Exception as exc:
            logger.warning(
                "unexpected error during MCP authentication",
                error_type=type(exc).__name__,
            )
            response = JSONResponse(
                {"error": {"code": 503, "message": "Service temporarily unavailable"}},
                status_code=503,
            )
            await response(scope, receive, send)
            return

        # Scope the session key to the authenticated user so one user cannot
        # present another user's mcp-session-id and hijack their workspace state.
        raw_sid = request.headers.get(_MCP_SESSION_HEADER) or ""
        session_id = f"{user.id}:{raw_sid}" if raw_sid else str(user.id)

        from src.mcp_server.session import (
            _current_session_id,
            get_or_create_session_state,
        )

        get_or_create_session_state(session_id, user)

        user_token = _current_user.set(user)
        sid_token = _current_session_id.set(session_id)
        try:
            await self._app(scope, receive, send)
        finally:
            _current_user.reset(user_token)
            _current_session_id.reset(sid_token)
