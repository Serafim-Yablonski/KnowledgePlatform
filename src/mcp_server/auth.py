"""MCP authentication: pure ASGI middleware + per-request ContextVar."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.composition import open_services
from src.core.exceptions import AppError, UnauthorizedError

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from src.models.user import User

# Carries the authenticated User for the duration of a single HTTP request.
_current_user: contextvars.ContextVar[User | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)

_MCP_SESSION_HEADER = "mcp-session-id"


def _looks_like_jwt(token: str) -> bool:
    """Return True if the token is structurally a JWT (header.payload.signature)."""
    return token.count(".") == 2


async def _authenticate_request(request: Request) -> User:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise UnauthorizedError(
            "Authentication required — provide Authorization: Bearer <token>"
        )
    token = auth_header[7:]
    async with open_services() as services:
        if _looks_like_jwt(token):
            # Definitely a JWT — surface the real error (expired, bad signature)
            # instead of masking it with "Invalid or revoked API key".
            return await services.auth.get_current_user(token)
        return await services.api_key.authenticate(token)


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
        except Exception as exc:  # noqa: BLE001 — top-level handler of the MCP ASGI app
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
