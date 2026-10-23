"""MCP server — protocol adapter served by the FastAPI application at /mcp."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp

from src.mcp_server.auth import MCPAuthMiddleware


def build_mcp_server() -> MCPServer:
    """The server with its instructions and every tool registered, not yet served.

    Instructions reach a client only in the server/discover result, and only some
    clients show them to their model, so they summarise the loop; each tool's
    description must stand on its own. The research tools stay out of them until
    Track 4 (a failed run reports "running" forever).
    """
    mcp = MCPServer(
        "knowledge-platform",
        instructions=(
            "Knowledge base of a team's documents, grouped into workspaces. Typical "
            "loop: list_workspaces, then search_documents to locate content (or "
            "list_documents to browse), then read_document around a hit, and search "
            "again if needed. Cite sources by document title and chunk_index."
        ),
    )

    from src.mcp_server.tools import register_tools

    register_tools(mcp)
    return mcp


def create_mcp_app() -> tuple[ASGIApp, StreamableHTTPSessionManager]:
    """Return the MCP Streamable HTTP handler behind auth middleware, and its manager.

    The caller serves the handler at its MCP path (``/mcp``). It is the session
    manager's own handler, not MCPServer's ``streamable_http_app()`` Starlette app:
    that app routes ``/mcp`` again, so the path would have to match in two places,
    and its error middleware would answer an SDK crash before ours does.

    The caller MUST enter the returned manager's run() in its app lifespan — the
    manager needs an anyio task group to handle requests, and run() works only once
    per instance, so every app gets its own manager.
    """
    mcp = build_mcp_server()

    # Creates the session manager. The SDK's DNS-rebinding guard is off: it can't
    # check Origin without also checking Host against an exact allowlist (every
    # unlisted hostname gets 421), and it runs after our middleware. MCPAuthMiddleware
    # checks Origin itself, first. Must be explicit: the default host (127.0.0.1)
    # switches the guard on.
    mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    manager = mcp.session_manager
    return MCPAuthMiddleware(manager.handle_request), manager
