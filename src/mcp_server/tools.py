"""MCP tools — thin wrappers over the service layer. No business logic here."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from src.core.database import get_session
from src.core.exceptions import InputValidationError
from src.core.http import get_async_http_client
from src.core.redis import get_async_redis_client
from src.mcp_server.auth import get_mcp_user

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal service factories (mirror src/core/dependencies.py without Depends)
# ---------------------------------------------------------------------------


def _make_workspace_service(session: Any) -> Any:
    from src.core.cache import ResponseCache
    from src.repositories.user import SQLAlchemyUserRepository
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository
    from src.repositories.workspace_cached import CachedWorkspaceRepository
    from src.services.workspace import WorkspaceService

    cache = ResponseCache(get_async_redis_client())
    inner_repo = SQLAlchemyWorkspaceRepository(session)
    return WorkspaceService(
        workspace_repo=CachedWorkspaceRepository(inner_repo, cache),
        user_repo=SQLAlchemyUserRepository(session),
    )


def _make_search_service(session: Any) -> Any:
    from src.ai.embeddings import EmbeddingService
    from src.core.config import get_settings
    from src.repositories.search import SQLAlchemySearchRepository
    from src.services.search import SearchService

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        http_client=get_async_http_client(),
    )
    return SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
    )


def _make_ai_service(session: Any) -> Any:
    from src.ai.embeddings import EmbeddingService
    from src.core.config import get_settings
    from src.repositories.document import SQLAlchemyDocumentRepository
    from src.repositories.search import SQLAlchemySearchRepository
    from src.services.ai import AIService
    from src.services.document import DocumentService
    from src.services.search import SearchService

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        http_client=get_async_http_client(),
    )
    search_svc = SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
    )
    return AIService(
        search_service=search_svc,
        document_service=DocumentService(
            repo=SQLAlchemyDocumentRepository(session), session=session
        ),
    )


def _make_research_service(session: Any, redis: Any) -> Any:
    from src.ai.embeddings import EmbeddingService
    from src.core.config import get_settings
    from src.repositories.search import SQLAlchemySearchRepository
    from src.services.research import ResearchService
    from src.services.search import SearchService

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        http_client=get_async_http_client(),
    )
    search_svc = SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
    )
    return ResearchService(search_service=search_svc, redis_client=redis)


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
    user = get_mcp_user()
    await _mcp_rate_limit(user.id, "mcp_search", 20, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        await workspace_svc.get_user_role(user, ws_uuid)  # ForbiddenError if not member
        search_svc = _make_search_service(session)
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
    user = get_mcp_user()
    await _mcp_rate_limit(user.id, "mcp_ask", 10, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        role = await workspace_svc.get_user_role(user, ws_uuid)
        ai_svc = _make_ai_service(session)
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
    user = get_mcp_user()
    await _mcp_rate_limit(user.id, "mcp_research_start", 5, 60)
    ws_uuid = _parse_workspace_id(workspace_id)
    redis = get_async_redis_client()
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        await workspace_svc.get_user_role(user, ws_uuid)
        research_svc = _make_research_service(session, redis)
        thread_id = await research_svc.start_research(
            workspace_id=ws_uuid,
            user_id=user.id,
            topic=topic,
            max_iterations=max_iterations,
        )
    return {"thread_id": thread_id, "status": "running"}


async def get_research_status(
    thread_id: Annotated[
        str,
        Field(description="Thread ID returned by start_research"),
    ],
    workspace_id: Annotated[
        str,
        Field(description="UUID of the workspace that owns this research thread"),
    ],
) -> dict[str, Any]:
    user = get_mcp_user()
    ws_uuid = _parse_workspace_id(workspace_id)
    redis = get_async_redis_client()
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        await workspace_svc.get_user_role(user, ws_uuid)
        research_svc = _make_research_service(session, redis)
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


# ---------------------------------------------------------------------------
# Session-scoped workspace tools
# ---------------------------------------------------------------------------


async def list_workspaces() -> list[dict[str, Any]]:
    from src.mcp_server.auth import get_mcp_user
    from src.mcp_server.session import (
        get_current_session_id,
        get_or_create_session_state,
    )

    user = get_mcp_user()
    session_id = get_current_session_id() or str(user.id)
    get_or_create_session_state(session_id, user)

    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
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


async def set_active_workspace(
    workspace_id: Annotated[
        str,
        Field(description="The workspace ID from list_workspaces"),
    ],
) -> dict[str, Any]:
    from src.mcp_server.auth import get_mcp_user
    from src.mcp_server.session import (
        get_current_session_id,
        get_or_create_session_state,
    )

    user = get_mcp_user()
    ws_uuid = _parse_workspace_id(workspace_id)
    session_id = get_current_session_id() or str(user.id)

    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        _, membership = await workspace_svc.get_workspace_for_user(ws_uuid, user.id)
        role = membership.role
        ws_info = await workspace_svc.get_by_id(ws_uuid)

    state = get_or_create_session_state(session_id, user)
    state.active_workspace_id = ws_uuid
    state.active_workspace_name = ws_info.name
    state.active_workspace_role = role

    return {
        "active_workspace_id": str(ws_uuid),
        "name": ws_info.name,
        "role": str(role),
        "member_count": ws_info.member_count,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_tools(mcp: FastMCP) -> None:
    mcp.add_tool(
        list_workspaces,
        description=(
            "List all workspaces you have access to. Call this first to discover "
            "available workspaces, then use set_active_workspace to select one. "
            "All subsequent search, ask, and research calls will use the active "
            "workspace."
        ),
    )
    mcp.add_tool(
        set_active_workspace,
        description=(
            "Set the active workspace for this session. Must be called after "
            "list_workspaces before any search, ask, or research operations."
        ),
    )
    mcp.add_tool(
        search_documents,
        description=(
            "Search the workspace's knowledge base for relevant document chunks. "
            "Returns ranked results with relevance scores."
        ),
    )
    mcp.add_tool(
        ask_question,
        description=(
            "Ask a question about the workspace's documents. Returns a structured "
            "answer with source citations and a confidence score."
        ),
    )
    mcp.add_tool(
        start_research,
        description=(
            "Start a multi-step research workflow that plans queries, retrieves "
            "evidence, evaluates sufficiency, and synthesises a report. "
            "Returns a thread_id to track progress."
        ),
    )
    mcp.add_tool(
        get_research_status,
        description=(
            "Check the status of a running research workflow. Returns current state "
            "including findings count and synthesis if complete."
        ),
    )
