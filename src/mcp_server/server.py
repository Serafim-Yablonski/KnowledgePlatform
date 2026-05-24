"""MCP server — protocol adapter mounted as a FastAPI sub-application at /mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp

from src.mcp_server.auth import MCPAuthMiddleware


def create_mcp_app() -> ASGIApp:
    """Return an ASGI app: FastMCP (Streamable HTTP) wrapped with auth middleware.

    Mounted at /mcp by the main FastAPI application. Shares the same database
    engine and Redis pool initialised during the main app's lifespan.
    """
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

    return MCPAuthMiddleware(mcp.streamable_http_app())
