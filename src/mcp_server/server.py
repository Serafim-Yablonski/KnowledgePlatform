"""MCP server — protocol adapter served by the FastAPI application at /mcp."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp

from src.mcp_server.auth import MCPAuthMiddleware


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
    mcp = MCPServer(
        "knowledge-platform",
        instructions=(
            "AI-powered knowledge base for engineering teams. "
            "Use search_documents to find relevant chunks, ask_question for "
            "direct Q&A with citations, and start_research / get_research_status "
            "for multi-step research workflows."
        ),
    )

    from src.mcp_server.tools import register_tools

    register_tools(mcp)

    # Creates the session manager. The SDK's DNS-rebinding guard is off: it ties the
    # spec's Origin check to a Host allowlist (exact hosts only — every real hostname
    # got 421 until configured) and runs after our middleware, so auth's 401 would
    # pre-empt the 403 the spec requires. MCPAuthMiddleware checks Origin first
    # instead. Must be explicit: the default host (127.0.0.1) switches the guard on.
    mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    manager = mcp.session_manager
    return MCPAuthMiddleware(manager.handle_request), manager
