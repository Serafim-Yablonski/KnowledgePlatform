"""MCP server — protocol adapter mounted as a FastAPI sub-application at /mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.types import ASGIApp

from src.mcp_server.auth import MCPAuthMiddleware

# Set during create_mcp_app() so the main lifespan can call session_manager.run().
_mcp_session_manager: StreamableHTTPSessionManager | None = None


def create_mcp_app() -> ASGIApp:
    """Return an ASGI app: FastMCP (Streamable HTTP) wrapped with auth middleware.

    Mounted at / by the main FastAPI application (FastMCP's inner route is /mcp,
    making the effective endpoint http://host/mcp). Shares the same database
    engine and Redis pool initialised during the main app's lifespan.

    The caller MUST enter get_mcp_session_manager().run() in the app lifespan —
    the StreamableHTTPSessionManager requires an anyio task group to handle requests.
    """
    global _mcp_session_manager

    mcp = FastMCP(
        "knowledge-platform",
        instructions=(
            "AI-powered knowledge base for engineering teams. "
            "Use search_documents to find relevant chunks, ask_question for "
            "direct Q&A with citations, and start_research / get_research_status "
            "for multi-step research workflows."
        ),
    )

    from src.mcp_server.resources import register_resources
    from src.mcp_server.tools import register_tools

    register_tools(mcp)
    register_resources(mcp)

    asgi_app = mcp.streamable_http_app()
    _mcp_session_manager = mcp.session_manager
    return MCPAuthMiddleware(asgi_app)


def get_mcp_session_manager() -> StreamableHTTPSessionManager:
    """Return the session manager created by create_mcp_app()."""
    if _mcp_session_manager is None:
        raise RuntimeError(
            "create_mcp_app() must be called before get_mcp_session_manager()"
        )
    return _mcp_session_manager
