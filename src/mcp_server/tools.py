"""MCP tools — thin wrappers over the service layer. No business logic here."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from src.composition import open_services
from src.core.exceptions import AppError, InputValidationError, RateLimitError
from src.core.redis import get_async_redis_client
from src.mcp_server.auth import McpContext, get_mcp_user


def _parse_workspace_id(workspace_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(workspace_id)
    except ValueError as exc:
        raise InputValidationError(f"Invalid workspace_id: {workspace_id!r}") from exc


async def _mcp_rate_limit(
    user_id: uuid.UUID, key_prefix: str, max_requests: int, window_seconds: int
) -> None:
    from src.core.rate_limit import SlidingWindowRateLimiter  # noqa: PLC0415

    limiter = SlidingWindowRateLimiter(
        get_async_redis_client(), key_prefix, max_requests, window_seconds
    )
    await limiter.check(str(user_id))


# ---------------------------------------------------------------------------
# Tool functions (module-level so tests can import and call them directly)
# ---------------------------------------------------------------------------


async def search_documents(
    ctx: McpContext,
    workspace_id: Annotated[str, Field(description="UUID of the workspace to search")],
    query: Annotated[
        str,
        Field(
            description="Natural language search query", min_length=1, max_length=2000
        ),
    ],
    top_k: Annotated[
        int,
        Field(description="Maximum number of results to return (1–20)", ge=1, le=20),
    ] = 5,
) -> list[dict[str, Any]]:
    user = get_mcp_user(ctx)
    await _mcp_rate_limit(user.id, "mcp_search", 20, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with open_services() as services:
        workspace_svc = services.workspace
        await workspace_svc.get_user_role(user, ws_uuid)  # ForbiddenError if not member
        search_svc = services.search
        result = await search_svc.search(workspace_id=ws_uuid, query=query, top_k=top_k)
    return [
        {
            "document_id": str(item.document_id),
            "document_title": item.document_title,
            "chunk_text": item.chunk_text,
            "score": round(item.score, 4),
            "metadata": item.chunk_metadata,
        }
        for item in result.results
    ]


async def ask_question(
    ctx: McpContext,
    workspace_id: Annotated[
        str, Field(description="UUID of the workspace whose documents to query")
    ],
    question: Annotated[
        str,
        Field(
            description="Question to answer using the workspace's documents",
            min_length=1,
            max_length=2000,
        ),
    ],
) -> dict[str, Any]:
    user = get_mcp_user(ctx)
    await _mcp_rate_limit(user.id, "mcp_ask", 10, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with open_services() as services:
        workspace_svc = services.workspace
        role = await workspace_svc.get_user_role(user, ws_uuid)
        ai_svc = services.ai
        answer = await ai_svc.ask(
            workspace_id=ws_uuid,
            user_id=user.id,
            question=question,
            role=role,
        )
    return {
        "answer": answer.answer,
        "confidence": answer.confidence,
        "reasoning": answer.reasoning,
        "sources": [
            {
                "document_id": str(s.document_id),
                "document_title": s.document_title,
                "chunk_text": s.chunk_text,
                "relevance_score": round(s.relevance_score, 4),
            }
            for s in answer.sources
        ],
    }


async def start_research(
    ctx: McpContext,
    workspace_id: Annotated[
        str, Field(description="UUID of the workspace to research")
    ],
    topic: Annotated[
        str,
        Field(
            description="Research topic or question to investigate",
            min_length=1,
            max_length=500,
        ),
    ],
    max_iterations: Annotated[
        int,
        Field(
            description="Maximum retrieval passes before synthesising (1–5)",
            ge=1,
            le=5,
        ),
    ] = 3,
) -> dict[str, Any]:
    user = get_mcp_user(ctx)
    await _mcp_rate_limit(user.id, "mcp_research_start", 5, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with open_services() as services:
        workspace_svc = services.workspace
        await workspace_svc.get_user_role(user, ws_uuid)
        research_svc = services.research
        thread_id = await research_svc.start_research(
            workspace_id=ws_uuid,
            user_id=user.id,
            topic=topic,
            max_iterations=max_iterations,
        )
    return {"thread_id": thread_id, "status": "running"}


async def get_research_status(
    ctx: McpContext,
    thread_id: Annotated[
        str,
        Field(description="Thread ID returned by start_research"),
    ],
    workspace_id: Annotated[
        str,
        Field(description="UUID of the workspace that owns this research thread"),
    ],
) -> dict[str, Any]:
    user = get_mcp_user(ctx)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with open_services() as services:
        workspace_svc = services.workspace
        await workspace_svc.get_user_role(user, ws_uuid)
        research_svc = services.research
        status = await research_svc.get_status(
            workspace_id=ws_uuid, user_id=user.id, thread_id=thread_id
        )
    result: dict[str, Any] = {
        "thread_id": status.thread_id,
        "status": status.status,
        "topic": status.topic,
        "findings_count": status.findings_count,
        "human_approved": status.human_approved,
        "synthesis": status.synthesis,
    }
    if status.plan is not None:
        result["plan"] = {
            "queries": status.plan.queries,
            "scope": status.plan.scope,
            "expected_sections": status.plan.expected_sections,
        }
    return result


async def list_workspaces(ctx: McpContext) -> list[dict[str, Any]]:
    user = get_mcp_user(ctx)
    async with open_services() as services:
        workspace_svc = services.workspace
        workspaces = await workspace_svc.list_for_user(user)
        results = []
        for ws in workspaces:
            role = await workspace_svc.get_user_role(user, ws.id)
            results.append(
                {
                    "id": str(ws.id),
                    "name": ws.name,
                    "slug": ws.slug,
                    "role": str(role),
                    "member_count": ws.member_count,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _app_errors_as_tool_errors(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Let the model read an AppError's detail; any other exception stays opaque.

    The SDK shows the model a ToolError's text but only "Error executing tool <name>"
    for anything else, so services' anticipated failures ("Not a member…") would be
    lost. ``detail`` is what REST already shows the same user. ``functools.wraps``
    keeps the signature the SDK builds the schema and finds ``ctx`` from.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except RateLimitError as exc:
            # REST sends the wait in Retry-After; the model has only this text.
            raise ToolError(
                f"{exc.detail} — retry in {exc.retry_after} seconds"
            ) from exc
        except AppError as exc:
            raise ToolError(exc.detail) from exc

    return wrapper


def _add_tool(
    mcp: MCPServer, fn: Callable[..., Awaitable[Any]], *, description: str
) -> None:
    mcp.add_tool(_app_errors_as_tool_errors(fn), description=description)


def register_tools(mcp: MCPServer) -> None:
    _add_tool(
        mcp,
        list_workspaces,
        description=(
            "List the workspaces you belong to, with your role in each. Call this "
            "first: every other tool takes a workspace id from here as workspace_id."
        ),
    )
    _add_tool(
        mcp,
        search_documents,
        description=(
            "Search the workspace's knowledge base for relevant document chunks. "
            "Returns ranked results with relevance scores."
        ),
    )
    _add_tool(
        mcp,
        ask_question,
        description=(
            "Ask a question about the workspace's documents. Returns a structured "
            "answer with source citations and a confidence score."
        ),
    )
    _add_tool(
        mcp,
        start_research,
        description=(
            "Start a multi-step research workflow that plans queries, retrieves "
            "evidence, evaluates sufficiency, and synthesises a report. "
            "Returns a thread_id to track progress."
        ),
    )
    _add_tool(
        mcp,
        get_research_status,
        description=(
            "Check the status of a running research workflow. Returns current state "
            "including findings count and synthesis if complete."
        ),
    )
