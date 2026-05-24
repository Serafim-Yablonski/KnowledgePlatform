# 012. MCP Server as a Protocol Adapter over the Shared Service Layer
Status: Accepted | Date: 2026-05-24

## Context
We need an MCP (Model Context Protocol) server so AI clients can search documents, ask questions, and run research workflows. The same business logic already exists behind the REST API. The key question is where to draw the boundary: should the MCP server duplicate service logic, access repositories directly, or share the existing service layer?

## Decision
The MCP server is a pure protocol adapter that imports exclusively from `services/`. It never imports from `repositories/`, `models/`, or the database layer directly. It shares the same authorization checks, caching, and business logic as the REST API:

```
REST API  ─┐
            ├─→ Service Layer → Repository → Database
MCP Server ─┘
```

The server is implemented as a FastMCP (Streamable HTTP) ASGI app mounted at `/mcp` on the same process as the REST API, sharing the lifespan-initialized database engine and Redis pool without an extra container.

Auth uses a pure ASGI middleware (not `BaseHTTPMiddleware`) that validates bearer JWTs and API keys before the MCP protocol layer sees the request. The authenticated user is propagated to tool functions via a `contextvars.ContextVar`, which Python propagates through async call chains automatically. This avoids buffering streaming SSE responses that `BaseHTTPMiddleware` would cause.

## Consequences
The service layer is the single source of truth. A bug fix or permission check added to `WorkspaceService.get_user_role` applies to both the REST API and MCP tools automatically, with no drift.

**Rejected: MCP server as a separate container with its own service implementations.** This would double the surface area for business logic, create two places to patch security bugs, and add an extra container + network hop for no architectural benefit.

**Rejected: MCP tools importing from `repositories/` directly.** This skips the service layer's authorization and caching, breaking the layering invariant that unit tests verify via Protocol stubs.

**Tradeoff:** Every tool call opens its own async session (mirrors FastAPI's per-request `get_db`). For high-throughput MCP usage this is fine; if session overhead becomes a bottleneck, connection pooling at the `async_sessionmaker` level is the lever to pull, not architectural change.
