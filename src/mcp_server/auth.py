"""MCP authentication: pure ASGI middleware + per-request ContextVar."""

from __future__ import annotations

import contextvars
import secrets
from typing import TYPE_CHECKING

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.core.config import settings
from src.core.database import get_session
from src.core.exceptions import AppError, ForbiddenError, UnauthorizedError
from src.repositories.user import SQLAlchemyUserRepository
from src.services.auth import AuthService

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from src.models.user import User

# Carries the authenticated User for the duration of a single HTTP request.
# Python propagates ContextVar values into child coroutines automatically, so
# tool functions called by FastMCP will see the value set here.
_current_user: contextvars.ContextVar[User | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)


async def _authenticate_request(request: Request) -> User:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        async with get_session() as session:
            repo = SQLAlchemyUserRepository(session)
            service = AuthService(repo)
            return await service.get_current_user(token)

    api_key = request.headers.get("X-API-Key", "")
    configured_key = settings.MCP_API_KEY
    if api_key and configured_key and secrets.compare_digest(api_key, configured_key):
        if not settings.MCP_API_KEY_USER_EMAIL:
            raise ForbiddenError("API key user email not configured")
        async with get_session() as session:
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
        except Exception:
            logger.warning("unexpected error during MCP authentication")
            response = JSONResponse(
                {"error": {"code": 503, "message": "Service temporarily unavailable"}},
                status_code=503,
            )
            await response(scope, receive, send)
            return

        token = _current_user.set(user)
        try:
            await self._app(scope, receive, send)
        finally:
            _current_user.reset(token)
