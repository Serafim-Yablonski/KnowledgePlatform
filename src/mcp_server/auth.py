"""MCP authentication: pure ASGI middleware, and the helper tools read the user with.

The middleware stores the user in the request's ASGI ``state``; the SDK hands each
tool the Request that carried its call, so identity follows every request. Not a
ContextVar: whether one follows each request depends on how the SDK carries context
between its tasks — ``mcp`` 1.x copied the session opener's into the session task,
so tools saw that user forever (F1). The Request is handed over explicitly.
"""

from __future__ import annotations

from typing import Any

import structlog
from mcp.server.mcpserver import Context
from mcp.shared.inbound import (
    ERROR_CODE_HTTP_STATUS,
    MCP_PROTOCOL_VERSION_HEADER,
    unsupported_protocol_version_rejection,
)
from mcp.types.jsonrpc import HEADER_MISMATCH, ErrorData, JSONRPCError
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from src.composition import open_services
from src.core.exceptions import AppError, UnauthorizedError
from src.models.user import User

logger = structlog.get_logger(__name__)

# What a tool's ``ctx`` parameter is annotated with. A runtime import, not under
# TYPE_CHECKING: MCPServer resolves the annotation to find the parameter to inject.
McpContext = Context[Any, Request]

_USER_STATE_KEY = "mcp_user"


def _looks_like_jwt(token: str) -> bool:
    """Return True if the token is structurally a JWT (header.payload.signature)."""
    return token.count(".") == 2


async def _authenticate_request(request: Request) -> User:
    # Parsed like REST's HTTPBearer: the scheme is case-insensitive (RFC 7235 §2.1).
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError(
            "Authentication required — provide Authorization: Bearer <token>"
        )
    async with open_services() as services:
        if _looks_like_jwt(token):
            # Can't be an API key (they never contain "."), so surface the JWT's real
            # error (expired, bad signature), not "Invalid or revoked API key".
            return await services.auth.get_current_user(token)
        return await services.api_key.authenticate(token)


def get_mcp_user(ctx: McpContext) -> User:
    """Return the user MCPAuthMiddleware authenticated for the request behind ``ctx``.

    Raises UnauthorizedError when there is no such request or user, so a tool reached
    any other way fails closed.
    """
    try:
        request = ctx.request_context.request
    except ValueError:  # Context created outside a request
        request = None
    user = (
        getattr(request.state, _USER_STATE_KEY, None) if request is not None else None
    )
    if not isinstance(user, User):
        raise UnauthorizedError("Authentication required")
    return user


def _refuse_legacy_protocol(request: Request) -> Response | None:
    """The spec's refusal of a pre-2026 request, or None to pass it on.

    The SDK routes on this header alone: present and not handshake-era goes to its
    2026 handler, anything else to its stateful legacy path, which we don't serve.
    Mirroring that rule exactly stops only legacy traffic; everything else (unknown
    versions, header/body mismatches) is left to the SDK. The body isn't read — its
    size limit is applied after us — so the error carries no request id.
    """
    version = request.headers.get(MCP_PROTOCOL_VERSION_HEADER)
    if version is not None and version not in HANDSHAKE_PROTOCOL_VERSIONS:
        return None
    if request.method != "POST":
        return Response(status_code=405, headers={"Allow": "POST"})
    if version is None:
        rejection = ErrorData(
            code=HEADER_MISMATCH,
            message=f"Missing required header: {MCP_PROTOCOL_VERSION_HEADER}",
        )
    else:
        unsupported = unsupported_protocol_version_rejection(version)
        assert unsupported is not None  # handshake-era versions are never served
        rejection = ErrorData(
            code=unsupported.code, message=unsupported.message, data=unsupported.data
        )
    error = JSONRPCError(jsonrpc="2.0", id=None, error=rejection)
    return JSONResponse(
        {**error.model_dump(mode="json", exclude_none=True), "id": None},
        status_code=ERROR_CODE_HTTP_STATUS[rejection.code],
    )


class MCPAuthMiddleware:
    """Pure ASGI middleware that rejects browser Origins, refuses pre-2026 protocol
    requests, then authenticates, before MCPServer sees the request.

    Uses a pure ASGI approach (not BaseHTTPMiddleware) to avoid buffering
    streaming SSE responses produced by the Streamable HTTP transport.
    Served by an exact HTTP ``Route``, so it only ever sees http scopes: the
    router closes websockets itself and never passes lifespan to a route.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope)
        # DNS rebinding: a malicious page makes the victim's browser call this
        # server. Browsers always send Origin on such requests and none of our
        # clients is a browser, so any Origin is refused. The bearer token already
        # stops the attack (a browser never attaches it by itself); this is the
        # spec's required check, kept in case ambient credentials ever appear.
        # First, so no 401 or 400 pre-empts the 403 the spec requires.
        if "origin" in request.headers:
            response = JSONResponse(
                {"error": {"code": 403, "message": "Origin not allowed"}},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        if (refusal := _refuse_legacy_protocol(request)) is not None:
            await refusal(scope, receive, send)
            return

        try:
            user = await _authenticate_request(request)
        except AppError as exc:
            response = JSONResponse(
                {"error": {"code": exc.status_code, "message": exc.detail}},
                status_code=exc.status_code,
                headers=(
                    {"WWW-Authenticate": exc.www_authenticate}
                    if isinstance(exc, UnauthorizedError)
                    else None
                ),
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

        # A copy of the server's ``state``, never the dict itself: the user can't leak
        # into another request even if a server shared that dict between requests.
        scope["state"] = {**scope.get("state", {}), _USER_STATE_KEY: user}
        await self._app(scope, receive, send)
