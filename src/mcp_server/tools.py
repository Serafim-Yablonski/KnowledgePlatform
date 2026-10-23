"""MCP tools — thin wrappers over the service layer. No business logic here."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import structlog
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from src.composition import open_services
from src.core.config import settings
from src.core.exceptions import AppError, RateLimitError
from src.core.rate_limit import SlidingWindowRateLimiter
from src.core.redis import get_async_redis_client
from src.domain.documents import DEFAULT_PAGE_SIZE, DocumentPageSize
from src.domain.search import SearchQuery
from src.mcp_server.auth import McpContext, get_mcp_user
from src.mcp_server.schemas import (
    ChunkContent,
    DocumentContent,
    DocumentList,
    DocumentSummary,
    SearchHit,
    SearchHits,
    WorkspaceList,
    WorkspaceSummary,
)

logger = structlog.get_logger(__name__)


async def _mcp_rate_limit(
    user_id: uuid.UUID, key_prefix: str, max_requests: int, window_seconds: int
) -> None:
    # REST's settings for the same operation, in MCP's own buckets (key_prefix).
    limiter = SlidingWindowRateLimiter(
        get_async_redis_client(),
        key_prefix,
        max_requests,
        window_seconds,
        fail_open=settings.RATE_LIMIT_FAIL_OPEN,
    )
    await limiter.check(str(user_id))


# ---------------------------------------------------------------------------
# Tool functions (module-level so tests can import and call them directly)
# ---------------------------------------------------------------------------


async def search_documents(
    ctx: McpContext,
    workspace_id: Annotated[
        uuid.UUID, Field(description="UUID of the workspace to search")
    ],
    query: Annotated[SearchQuery, Field(description="Natural language search query")],
    top_k: Annotated[
        int,
        Field(description="Maximum number of results to return (1–20)", ge=1, le=20),
    ] = 5,
) -> SearchHits:
    user = get_mcp_user(ctx)
    await _mcp_rate_limit(
        user.id,
        "mcp_search",
        settings.RATE_LIMIT_SEARCH_REQUESTS,
        settings.RATE_LIMIT_SEARCH_WINDOW,
    )
    async with open_services() as services:
        workspace_svc = services.workspace
        # ForbiddenError if not a member
        await workspace_svc.get_user_role(user, workspace_id)
        search_svc = services.search
        result = await search_svc.search(
            workspace_id=workspace_id, query=query, top_k=top_k
        )
    return SearchHits(
        results=[
            SearchHit(
                document_id=item.document_id,
                document_title=item.document_title,
                chunk_index=item.chunk_index,
                section=item.section,
                chunk_text=item.chunk_text,
                score=round(item.score, 4),
            )
            for item in result.results
        ]
    )


async def list_documents(
    ctx: McpContext,
    workspace_id: Annotated[
        uuid.UUID, Field(description="UUID of the workspace to list")
    ],
    cursor: Annotated[
        str | None,
        Field(description="next_cursor from the previous page; omit for the first"),
    ] = None,
    limit: Annotated[
        DocumentPageSize, Field(description="Documents per page (1–100)")
    ] = DEFAULT_PAGE_SIZE,
) -> DocumentList:
    user = get_mcp_user(ctx)
    async with open_services() as services:
        # ForbiddenError if not a member
        await services.workspace.get_user_role(user, workspace_id)
        page = await services.document.list(workspace_id, cursor, limit)
    return DocumentList(
        documents=[
            DocumentSummary(
                id=doc.id,
                title=doc.title,
                content_type=doc.content_type,
                status=doc.status,
                created_at=doc.created_at,
            )
            for doc in page.items
        ],
        next_cursor=page.next_cursor,
    )


async def read_document(
    ctx: McpContext,
    workspace_id: Annotated[
        uuid.UUID, Field(description="UUID of the workspace that holds the document")
    ],
    document_id: Annotated[
        uuid.UUID, Field(description="UUID of the document to read")
    ],
    start_chunk: Annotated[
        int, Field(description="chunk_index to start from (0 = beginning)", ge=0)
    ] = 0,
    max_chunks: Annotated[
        int, Field(description="Maximum number of chunks to return (1–20)", ge=1, le=20)
    ] = 8,
) -> DocumentContent:
    user = get_mcp_user(ctx)
    async with open_services() as services:
        # ForbiddenError if not a member
        await services.workspace.get_user_role(user, workspace_id)
        doc = await services.document.read(
            workspace_id, document_id, start_chunk, max_chunks
        )
    return DocumentContent(
        document_id=doc.document_id,
        title=doc.title,
        total_chunks=doc.total_chunks,
        chunks=[
            ChunkContent(
                chunk_index=c.chunk_index, section=c.section, chunk_text=c.chunk_text
            )
            for c in doc.chunks
        ],
        next_start_chunk=doc.next_start_chunk,
    )


async def start_research(
    ctx: McpContext,
    workspace_id: Annotated[
        uuid.UUID, Field(description="UUID of the workspace to research")
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
    await _mcp_rate_limit(
        user.id,
        "mcp_research_start",
        settings.RATE_LIMIT_RESEARCH_START_REQUESTS,
        settings.RATE_LIMIT_RESEARCH_START_WINDOW,
    )
    async with open_services() as services:
        workspace_svc = services.workspace
        await workspace_svc.get_user_role(user, workspace_id)
        research_svc = services.research
        thread_id = await research_svc.start_research(
            workspace_id=workspace_id,
            user_id=user.id,
            topic=topic,
            max_iterations=max_iterations,
        )
    return {"thread_id": thread_id, "status": "running"}


async def get_research_status(
    ctx: McpContext,
    thread_id: Annotated[
        uuid.UUID,
        Field(description="Thread ID returned by start_research"),
    ],
    workspace_id: Annotated[
        uuid.UUID,
        Field(description="UUID of the workspace that owns this research thread"),
    ],
) -> dict[str, Any]:
    user = get_mcp_user(ctx)
    async with open_services() as services:
        workspace_svc = services.workspace
        await workspace_svc.get_user_role(user, workspace_id)
        research_svc = services.research
        status = await research_svc.get_status(
            workspace_id=workspace_id, user_id=user.id, thread_id=str(thread_id)
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


async def list_workspaces(ctx: McpContext) -> WorkspaceList:
    user = get_mcp_user(ctx)
    async with open_services() as services:
        workspaces = await services.workspace.list_for_user(user)
    return WorkspaceList(
        workspaces=[
            WorkspaceSummary(id=ws.id, name=ws.name, description=ws.description)
            for ws in workspaces
        ]
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _app_errors_as_tool_errors(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Let the model read an AppError's detail; any other exception stays opaque.

    The SDK shows the model a ToolError's text but only "Error executing tool <name>"
    for anything else, so services' anticipated failures ("Not a member…") would be
    lost. ``detail`` is what REST already shows the same user. As in REST, a 5xx is
    the server's fault and is logged at ERROR; the SDK logs every tool error at INFO.
    ``functools.wraps`` keeps the signature the SDK builds the schema and finds
    ``ctx`` from.
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
            if exc.status_code >= 500:
                logger.error(
                    "tool error",
                    tool=fn.__name__,
                    error_type=type(exc).__name__,
                    status_code=exc.status_code,
                )
            raise ToolError(exc.detail) from exc

    return wrapper


def _add_tool(
    mcp: MCPServer,
    fn: Callable[..., Awaitable[Any]],
    *,
    description: str,
    read_only: bool,
) -> None:
    """Register ``fn`` behind the error wrapper, declaring its effects.

    Clients base approval policy on these spec hints (e.g. Claude Code's plan mode
    runs only read-only tools). Every tool here reads or adds to our own corpus, so
    none is open-world or destructive — both spec defaults.
    """
    annotations = ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=None if read_only else False,
        open_world_hint=False,
    )
    mcp.add_tool(
        _app_errors_as_tool_errors(fn), description=description, annotations=annotations
    )


def register_tools(mcp: MCPServer) -> None:
    _add_tool(
        mcp,
        list_workspaces,
        read_only=True,
        description=(
            "List the workspaces you can read: id, name and description. Start "
            "here: every other tool takes a workspace_id from this list."
        ),
    )
    _add_tool(
        mcp,
        search_documents,
        read_only=True,
        description=(
            "Semantic (meaning-based) search over a workspace's indexed documents. "
            "Returns up to top_k chunks, best first, each with document_id, "
            "document_title, chunk_index, section, chunk_text and a 0-1 score. Exact "
            "strings such as error codes or ids may be missed. To see more around a "
            "hit, call read_document with its document_id and a start_chunk a "
            "little before its chunk_index."
        ),
    )
    _add_tool(
        mcp,
        list_documents,
        read_only=True,
        description=(
            "List a workspace's documents, newest first: id, title, content_type, "
            "status and created_at. Pass next_cursor back as cursor for the next "
            "page; null means the last page. Only indexed documents can be searched "
            "or read. Use this to browse or find a document by title; use "
            "search_documents to find content."
        ),
    )
    _add_tool(
        mcp,
        read_document,
        read_only=True,
        description=(
            "Read an indexed document's text in order, a few chunks at a time: "
            "after search_documents to read around a hit, or from the start. "
            "Returns title, total_chunks, the chunks (chunk_index, section, "
            "chunk_text) and next_start_chunk: pass it as start_chunk to continue; "
            "null means the end. Neighbouring chunks repeat a little text at the "
            "seams."
        ),
    )
    _add_tool(
        mcp,
        start_research,
        read_only=False,
        description=(
            "Start a multi-step research workflow that plans queries, retrieves "
            "evidence, evaluates sufficiency, and synthesises a report. "
            "Returns a thread_id to track progress."
        ),
    )
    _add_tool(
        mcp,
        get_research_status,
        read_only=True,
        description=(
            "Check the status of a running research workflow. Returns current state "
            "including findings count and synthesis if complete."
        ),
    )
