"""Unit tests for MCP tools and auth helpers.

Services are mocked via Protocol stubs — no DB or Redis required.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import structlog
from fakeredis.aioredis import FakeRedis
from fastapi.security.utils import get_authorization_scheme_param
from mcp import Client
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.requests import Request

import src.mcp_server.tools
from src.core.config import settings
from src.core.exceptions import (
    AppError,
    ForbiddenError,
    InputValidationError,
    InvalidTokenError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from src.core.redis import PREFIX_RATELIMIT
from src.domain.documents import (
    ChunkText,
    ContentType,
    DocumentChunks,
    DocumentPage,
    DocumentStatus,
)
from src.domain.roles import WorkspaceRole
from src.domain.search import SearchResult, SearchResults
from src.domain.workspace import WorkspaceInfo
from src.mcp_server.auth import McpContext, _authenticate_request, get_mcp_user
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
from src.mcp_server.server import build_mcp_server
from src.mcp_server.tools import (
    get_research_status,
    list_documents,
    list_workspaces,
    read_document,
    register_tools,
    search_documents,
    start_research,
)
from src.models.document import Document
from src.models.user import User
from src.schemas.research import ResearchPlanResponse, ResearchStatusResponse
from src.services.document import INVALID_CURSOR_DETAIL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_user(user_id: uuid.UUID | None = None) -> User:
    # A real (transient) User: get_mcp_user checks the type.
    return User(id=user_id or uuid.uuid4(), email="user@example.com", is_active=True)


def _ctx_with_state(
    state: dict[str, Any] | None, *, with_request: bool = True
) -> Context[Any, Any]:
    """A tool Context over a real Starlette Request, as MCPServer builds it."""
    scope: dict[str, Any] = {"type": "http", "method": "POST", "path": "/mcp"}
    if state is not None:
        scope["state"] = state
    request_context: ServerRequestContext[Any, Any] = ServerRequestContext(
        request_id=1,
        protocol_version="2026-07-28",
        method="tools/call",
        meta=None,
        session=MagicMock(),
        lifespan_context=None,
        request=Request(scope) if with_request else None,
    )
    return Context(request_context=request_context)


def _ctx(user: User | None, *, with_request: bool = True) -> Context[Any, Any]:
    """The Context of a request the MCP auth middleware let through as ``user``."""
    state = None if user is None else {"mcp_user": user}
    return _ctx_with_state(state, with_request=with_request)


def _make_search_response(n: int = 1) -> SearchResults:
    doc_id = uuid.uuid4()
    return SearchResults(
        results=[
            SearchResult(
                chunk_id=uuid.uuid4(),
                chunk_index=i,
                chunk_text="relevant text",
                document_id=doc_id,
                document_title="Test Doc",
                score=0.9,
                section=None,
            )
            for i in range(n)
        ],
        query="test query",
        total_results=n,
    )


@pytest.fixture
def mock_user() -> User:
    return _make_user()


@pytest.fixture
def fake_services():
    """Patch open_services to yield a stub Services; tests attach service mocks."""
    services = MagicMock()

    @asynccontextmanager
    async def _fake_open_services() -> AsyncGenerator[MagicMock]:
        yield services

    with patch("src.mcp_server.tools.open_services", _fake_open_services):
        yield services


@pytest.fixture
def mock_redis():
    redis_mock = MagicMock()
    pipe_mock = MagicMock()
    # pipeline() is sync; only execute() is awaited. Return count=1 so the rate
    # limiter always passes in tests (well under any per-user limit).
    pipe_mock.execute = AsyncMock(return_value=[1, 0, 1, [], 1])
    redis_mock.pipeline.return_value = pipe_mock
    with patch("src.mcp_server.tools.get_async_redis_client", return_value=redis_mock):
        yield


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestGetMcpUser:
    def test_returns_user_from_request_state(self) -> None:
        user = _make_user()
        assert get_mcp_user(_ctx(user)) is user

    def test_raises_outside_a_request(self) -> None:
        # Context() without a request context raises ValueError on access.
        with pytest.raises(UnauthorizedError):
            get_mcp_user(Context())

    def test_raises_when_request_missing(self) -> None:
        with pytest.raises(UnauthorizedError):
            get_mcp_user(_ctx(None, with_request=False))

    @pytest.mark.parametrize("state", [{}, {"mcp_user": "alice"}])
    def test_raises_when_state_holds_no_user(self, state: dict[str, Any]) -> None:
        with pytest.raises(UnauthorizedError):
            get_mcp_user(_ctx_with_state(state))


def _workspace_info(name: str, description: str | None) -> WorkspaceInfo:
    return WorkspaceInfo(
        id=uuid.uuid4(),
        name=name,
        slug=name.lower(),
        description=description,
        is_active=True,
        created_at=datetime(2026, 1, 1),
        member_count=3,
    )


class TestListWorkspacesTool:
    async def test_returns_what_identifies_and_describes_each_workspace(
        self, fake_services: MagicMock
    ) -> None:
        # The id for other tools, and what helps the model pick one; role, slug and
        # member_count change nothing a tool does.
        user = _make_user()
        infos = [_workspace_info("Infra", "Runbooks"), _workspace_info("ML", None)]
        fake_services.workspace.list_for_user = AsyncMock(return_value=infos)

        result = await list_workspaces(ctx=_ctx(user))

        fake_services.workspace.list_for_user.assert_awaited_once_with(user)
        assert result == WorkspaceList(
            workspaces=[
                WorkspaceSummary(
                    id=info.id, name=info.name, description=info.description
                )
                for info in infos
            ]
        )
        assert set(WorkspaceSummary.model_fields) == {"id", "name", "description"}


def _document(title: str, status: DocumentStatus) -> Document:
    return Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        title=title,
        content_type=ContentType.MARKDOWN,
        status=status,
        file_path="/tmp/x.md",
        file_size_bytes=100,
        uploaded_by=uuid.uuid4(),
        version=1,
        created_at=datetime(2026, 10, 22, 12, 0, 0),
        updated_at=datetime(2026, 10, 22, 12, 0, 0),
    )


class TestListDocumentsTool:
    async def test_returns_a_page_of_summaries_and_the_next_cursor(
        self, fake_services: MagicMock
    ) -> None:
        # What lets the model pick a document and see whether it is searchable yet;
        # size, uploader and version change nothing a tool does.
        user = _make_user()
        ws_id = uuid.uuid4()
        docs = [
            _document("Runbook", DocumentStatus.INDEXED),
            _document("Draft", DocumentStatus.PROCESSING),
        ]
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )
        fake_services.document.list = AsyncMock(
            return_value=DocumentPage(items=docs, next_cursor="next")
        )

        result = await list_documents(
            ctx=_ctx(user), workspace_id=ws_id, cursor="previous", limit=2
        )

        fake_services.workspace.get_user_role.assert_awaited_once_with(user, ws_id)
        fake_services.document.list.assert_awaited_once_with(ws_id, "previous", 2)
        assert result == DocumentList(
            documents=[
                DocumentSummary(
                    id=d.id,
                    title=d.title,
                    content_type=d.content_type,
                    status=d.status,
                    created_at=d.created_at,
                )
                for d in docs
            ],
            next_cursor="next",
        )
        assert set(DocumentSummary.model_fields) == {
            "id",
            "title",
            "content_type",
            "status",
            "created_at",
        }

    async def test_invalid_cursor_tells_the_model_how_to_recover(
        self, mock_user: User, fake_services: MagicMock
    ) -> None:
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )
        fake_services.document.list = AsyncMock(
            side_effect=InputValidationError(INVALID_CURSOR_DETAIL)
        )
        mcp = MCPServer("test")
        register_tools(mcp)

        with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "list_documents",
                    {"workspace_id": str(uuid.uuid4()), "cursor": "truncated"},
                )

        assert _error_text(result) == (
            f"Error executing tool list_documents: {INVALID_CURSOR_DETAIL}"
        )

    # The same bounds as REST (shared DocumentPageSize); 0 would crash the query.
    @pytest.mark.parametrize(
        ("args", "limit"),
        [
            pytest.param({}, 20, id="default"),
            pytest.param({"limit": 1}, 1, id="min"),
            pytest.param({"limit": 100}, 100, id="max"),
            pytest.param({"limit": 0}, None, id="zero"),
            pytest.param({"limit": 101}, None, id="over-max"),
        ],
    )
    async def test_limit_is_bounded_before_the_tool_runs(
        self,
        args: dict[str, Any],
        limit: int | None,
        mock_user: User,
        fake_services: MagicMock,
    ) -> None:
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )
        fake_services.document.list = AsyncMock(
            return_value=DocumentPage(items=[], next_cursor=None)
        )
        mcp = MCPServer("test")
        register_tools(mcp)
        ws_id = uuid.uuid4()

        with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "list_documents", {"workspace_id": str(ws_id), **args}
                )

        if limit is None:
            assert "limit" in _error_text(result)
            fake_services.workspace.get_user_role.assert_not_called()
        else:
            assert not result.is_error, result.content
            fake_services.document.list.assert_awaited_once_with(ws_id, None, limit)

    async def test_limit_bounds_are_in_the_schema(self) -> None:
        schema = (await _input_schemas())["list_documents"]["properties"]["limit"]

        assert (schema["minimum"], schema["maximum"], schema["default"]) == (1, 100, 20)


def _document_chunks(document_id: uuid.UUID) -> DocumentChunks:
    return DocumentChunks(
        document_id=document_id,
        title="Runbook",
        total_chunks=40,
        chunks=[
            ChunkText(chunk_index=11, section="## Deploy", chunk_text="eleven"),
            ChunkText(chunk_index=12, section=None, chunk_text="twelve"),
        ],
        next_start_chunk=13,
    )


class TestReadDocumentTool:
    async def test_returns_the_chunks_and_where_to_read_on(
        self, fake_services: MagicMock
    ) -> None:
        user = _make_user()
        ws_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )
        fake_services.document.read = AsyncMock(return_value=_document_chunks(doc_id))

        result = await read_document(
            ctx=_ctx(user),
            workspace_id=ws_id,
            document_id=doc_id,
            start_chunk=11,
            max_chunks=2,
        )

        fake_services.workspace.get_user_role.assert_awaited_once_with(user, ws_id)
        fake_services.document.read.assert_awaited_once_with(ws_id, doc_id, 11, 2)
        assert result == DocumentContent(
            document_id=doc_id,
            title="Runbook",
            total_chunks=40,
            chunks=[
                ChunkContent(chunk_index=11, section="## Deploy", chunk_text="eleven"),
                ChunkContent(chunk_index=12, section=None, chunk_text="twelve"),
            ],
            next_start_chunk=13,
        )
        assert set(DocumentContent.model_fields) == {
            "document_id",
            "title",
            "total_chunks",
            "chunks",
            "next_start_chunk",
        }
        assert set(ChunkContent.model_fields) == {
            "chunk_index",
            "section",
            "chunk_text",
        }

    @pytest.mark.parametrize(
        ("args", "forwarded"),
        [
            pytest.param({}, (0, 8), id="defaults"),
            pytest.param({"start_chunk": 0, "max_chunks": 1}, (0, 1), id="min"),
            pytest.param({"max_chunks": 20}, (0, 20), id="max"),
            pytest.param({"start_chunk": -1}, None, id="negative-start"),
            pytest.param({"max_chunks": 0}, None, id="zero-chunks"),
            pytest.param({"max_chunks": 21}, None, id="over-max"),
        ],
    )
    async def test_arguments_are_bounded_before_the_tool_runs(
        self,
        args: dict[str, Any],
        forwarded: tuple[int, int] | None,
        mock_user: User,
        fake_services: MagicMock,
    ) -> None:
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )
        ws_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fake_services.document.read = AsyncMock(return_value=_document_chunks(doc_id))
        mcp = MCPServer("test")
        register_tools(mcp)

        with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "read_document",
                    {"workspace_id": str(ws_id), "document_id": str(doc_id), **args},
                )

        if forwarded is None:
            assert next(iter(args)) in _error_text(result)
            fake_services.workspace.get_user_role.assert_not_called()
        else:
            assert not result.is_error, result.content
            fake_services.document.read.assert_awaited_once_with(
                ws_id, doc_id, *forwarded
            )

    async def test_argument_bounds_are_in_the_schema(self) -> None:
        props = (await _input_schemas())["read_document"]["properties"]

        assert (props["start_chunk"]["minimum"], props["start_chunk"]["default"]) == (
            0,
            0,
        )
        assert (
            props["max_chunks"]["minimum"],
            props["max_chunks"]["maximum"],
            props["max_chunks"]["default"],
        ) == (1, 20, 8)
        assert props["document_id"]["format"] == "uuid"

    async def test_malformed_document_id_is_refused_before_the_tool_runs(
        self, mock_user: User, fake_services: MagicMock
    ) -> None:
        mcp = MCPServer("test")
        register_tools(mcp)

        with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "read_document",
                    {"workspace_id": str(uuid.uuid4()), "document_id": "chunk-12"},
                )

        assert "document_id" in _error_text(result)
        fake_services.workspace.get_user_role.assert_not_called()


async def test_no_tool_exposes_ctx_to_the_model() -> None:
    mcp = MCPServer("test")
    register_tools(mcp)

    tools = await mcp.list_tools()

    assert tools
    for tool in tools:
        assert "ctx" not in tool.input_schema.get("properties", {}), tool.name


@pytest.fixture
def auth_services():
    """Patch open_services in the auth module; tests configure auth/api_key mocks."""
    services = MagicMock()

    @asynccontextmanager
    async def _fake_open_services() -> AsyncGenerator[MagicMock]:
        yield services

    with patch("src.mcp_server.auth.open_services", _fake_open_services):
        yield services


class TestAuthenticateRequest:
    async def test_jwt_bearer_uses_auth_service(self, auth_services: MagicMock) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer header.payload.signature"}
        expected_user = _make_user()
        auth_services.auth.get_current_user = AsyncMock(return_value=expected_user)
        auth_services.api_key.authenticate = AsyncMock()

        result = await _authenticate_request(request)

        assert result is expected_user
        auth_services.auth.get_current_user.assert_awaited_once_with(
            "header.payload.signature"
        )
        auth_services.api_key.authenticate.assert_not_awaited()

    async def test_opaque_bearer_uses_api_key_service(
        self, auth_services: MagicMock
    ) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer not-a-jwt-no-dots"}
        expected_user = _make_user()
        auth_services.api_key.authenticate = AsyncMock(return_value=expected_user)

        result = await _authenticate_request(request)

        assert result is expected_user
        auth_services.api_key.authenticate.assert_awaited_once_with("not-a-jwt-no-dots")

    @pytest.mark.parametrize(
        "header",
        ["Bearer  padded-key", "Bearer padded-key ", "bearer \tpadded-key"],
    )
    async def test_token_parsed_like_rest(
        self, auth_services: MagicMock, header: str
    ) -> None:
        # The docstring's claim: the same header yields the same token on REST and MCP.
        request = MagicMock()
        request.headers = {"Authorization": header}
        auth_services.api_key.authenticate = AsyncMock(return_value=_make_user())

        await _authenticate_request(request)

        _, rest_token = get_authorization_scheme_param(header)
        auth_services.api_key.authenticate.assert_awaited_once_with(rest_token)

    async def test_no_credentials_raises(self) -> None:
        request = MagicMock()
        request.headers = {}

        with pytest.raises(UnauthorizedError, match="Authentication required"):
            await _authenticate_request(request)

    async def test_whitespace_only_token_is_no_credentials(self) -> None:
        # Not InvalidTokenError: nothing was sent, so the challenge carries no error.
        request = MagicMock()
        request.headers = {"Authorization": "Bearer   "}

        with pytest.raises(UnauthorizedError) as exc_info:
            await _authenticate_request(request)
        assert exc_info.type is UnauthorizedError

    async def test_revoked_api_key_raises_forbidden(
        self, auth_services: MagicMock
    ) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer not-a-jwt-no-dots"}
        auth_services.api_key.authenticate = AsyncMock(
            side_effect=InvalidTokenError("Invalid or revoked API key")
        )

        with pytest.raises(InvalidTokenError, match="Invalid or revoked"):
            await _authenticate_request(request)

    async def test_db_error_propagates(self) -> None:
        """A DB error during auth propagates rather than being masked as 401."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer not-a-jwt-no-dots"}

        @asynccontextmanager
        async def _failing() -> AsyncGenerator[MagicMock]:
            raise RuntimeError("DB unavailable")
            yield  # noqa: RET508

        with (
            patch("src.mcp_server.auth.open_services", _failing),
            pytest.raises(RuntimeError, match="DB unavailable"),
        ):
            await _authenticate_request(request)


# ---------------------------------------------------------------------------
# Tool: search_documents
# ---------------------------------------------------------------------------


class TestSearchDocumentsTool:
    async def test_calls_search_service(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        search_response = _make_search_response(2)

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_search_svc = MagicMock()
        mock_search_svc.search = AsyncMock(return_value=search_response)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "search", mock_search_svc),
        ):
            result = await search_documents(
                _ctx(mock_user), workspace_id=ws_id, query="test query", top_k=3
            )

        mock_search_svc.search.assert_awaited_once_with(
            workspace_id=ws_id, query="test query", top_k=3
        )
        assert len(result.results) == 2
        assert result.results[0].document_title == "Test Doc"

    async def test_rejects_unauthenticated(self) -> None:
        with pytest.raises(UnauthorizedError):
            await search_documents(_ctx(None), workspace_id=uuid.uuid4(), query="q")

    async def test_rejects_non_member(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(
            side_effect=ForbiddenError("Not a member")
        )

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            pytest.raises(ForbiddenError),
        ):
            await search_documents(
                _ctx(mock_user), workspace_id=uuid.uuid4(), query="q"
            )

    async def test_formats_results_correctly(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        doc_id = uuid.uuid4()
        search_response = SearchResults(
            results=[
                SearchResult(
                    chunk_id=uuid.uuid4(),
                    chunk_index=12,
                    chunk_text="chunk content",
                    document_id=doc_id,
                    document_title="My Document",
                    score=0.87654,
                    section="Setup > Install",
                )
            ],
            query="q",
            total_results=1,
        )

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.VIEWER)
        mock_search_svc = MagicMock()
        mock_search_svc.search = AsyncMock(return_value=search_response)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "search", mock_search_svc),
        ):
            result = await search_documents(
                _ctx(mock_user), workspace_id=uuid.uuid4(), query="q"
            )

        assert result == SearchHits(
            results=[
                SearchHit(
                    document_id=doc_id,
                    document_title="My Document",
                    chunk_index=12,
                    section="Setup > Install",
                    chunk_text="chunk content",
                    score=0.8765,  # rounded to 4 dp
                )
            ]
        )
        # The same hit fields as REST; no raw chunk metadata.
        assert set(SearchHit.model_fields) == {
            "document_id",
            "document_title",
            "chunk_index",
            "section",
            "chunk_text",
            "score",
        }


# (tool, REST's settings for the same operation, MCP's bucket)
_LIMITED_TOOLS = [
    pytest.param(
        search_documents,
        "RATE_LIMIT_SEARCH",
        "mcp_search",
        id="search_documents",
    ),
    pytest.param(
        start_research,
        "RATE_LIMIT_RESEARCH_START",
        "mcp_research_start",
        id="start_research",
    ),
]


class TestRateLimitedTools:
    """REST's settings for the same operation, own buckets (abuse plan D2)."""

    @pytest.fixture
    def member(self, fake_services: MagicMock) -> MagicMock:
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.MEMBER
        )
        fake_services.search.search = AsyncMock(return_value=_make_search_response(0))
        fake_services.research.start_research = AsyncMock(
            return_value=str(uuid.uuid4())
        )
        return fake_services

    @pytest.mark.parametrize(("tool", "setting", "bucket"), _LIMITED_TOOLS)
    async def test_limit_follows_the_rest_setting(
        self,
        tool: Any,
        setting: str,
        bucket: str,
        mock_user: User,
        member: MagicMock,
        settings_override: pytest.MonkeyPatch,
    ) -> None:
        settings_override.setattr(settings, f"{setting}_REQUESTS", 2)
        settings_override.setattr(settings, f"{setting}_WINDOW", 60)
        redis = FakeRedis()

        with patch("src.mcp_server.tools.get_async_redis_client", return_value=redis):
            for _ in range(2):
                await tool(_ctx(mock_user), uuid.uuid4(), "q")
            with pytest.raises(RateLimitError):
                await tool(_ctx(mock_user), uuid.uuid4(), "q")

        assert await redis.keys() == [
            f"{PREFIX_RATELIMIT}:{bucket}:{mock_user.id}".encode()
        ]
        await redis.aclose()

    @pytest.mark.parametrize("fail_open", [True, False])
    async def test_redis_failure_follows_the_fail_open_setting(
        self,
        fail_open: bool,
        mock_user: User,
        member: MagicMock,
        settings_override: pytest.MonkeyPatch,
    ) -> None:
        settings_override.setattr(settings, "RATE_LIMIT_FAIL_OPEN", fail_open)
        redis = MagicMock()
        redis.pipeline.return_value.execute = AsyncMock(
            side_effect=RedisConnectionError("down")
        )

        with patch("src.mcp_server.tools.get_async_redis_client", return_value=redis):
            if fail_open:
                await search_documents(_ctx(mock_user), uuid.uuid4(), "q")
            else:
                with pytest.raises(ServiceUnavailableError):
                    await search_documents(_ctx(mock_user), uuid.uuid4(), "q")

        assert member.search.search.await_count == int(fail_open)


# ---------------------------------------------------------------------------
# Tool: start_research
# ---------------------------------------------------------------------------


class TestStartResearchTool:
    async def test_calls_research_service(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        thread_id = str(uuid.uuid4())

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_research_svc = MagicMock()
        mock_research_svc.start_research = AsyncMock(return_value=thread_id)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "research", mock_research_svc),
        ):
            result = await start_research(
                _ctx(mock_user),
                workspace_id=ws_id,
                topic="AI trends",
                max_iterations=2,
            )

        mock_research_svc.start_research.assert_awaited_once_with(
            workspace_id=ws_id,
            user_id=mock_user.id,
            topic="AI trends",
            max_iterations=2,
        )
        assert result["thread_id"] == thread_id
        assert result["status"] == "running"


# ---------------------------------------------------------------------------
# Tool: get_research_status
# ---------------------------------------------------------------------------


class TestGetResearchStatusTool:
    async def test_returns_status_dict(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        thread_id = str(uuid.uuid4())
        status = ResearchStatusResponse(
            thread_id=thread_id,
            status="completed",
            topic="AI trends",
            plan=ResearchPlanResponse(
                queries=["q1"], scope="broad", expected_sections=["intro"]
            ),
            findings_count=5,
            synthesis="The answer is ...",
            human_approved=True,
        )

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_research_svc = MagicMock()
        mock_research_svc.get_status = AsyncMock(return_value=status)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "research", mock_research_svc),
        ):
            result = await get_research_status(
                _ctx(mock_user), thread_id=uuid.UUID(thread_id), workspace_id=ws_id
            )

        assert result["thread_id"] == thread_id
        assert result["status"] == "completed"
        assert result["synthesis"] == "The answer is ..."
        assert result["plan"]["queries"] == ["q1"]
        assert result["human_approved"] is True

    async def test_omits_plan_when_none(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        thread_id = str(uuid.uuid4())
        status = ResearchStatusResponse(
            thread_id=thread_id,
            status="running",
            topic="topic",
            plan=None,
            findings_count=0,
            synthesis=None,
        )

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_research_svc = MagicMock()
        mock_research_svc.get_status = AsyncMock(return_value=status)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "research", mock_research_svc),
        ):
            result = await get_research_status(
                _ctx(mock_user), thread_id=uuid.UUID(thread_id), workspace_id=ws_id
            )

        assert "plan" not in result
        assert result["synthesis"] is None


# ---------------------------------------------------------------------------
# Tool errors: what the model reads when a registered tool fails
# ---------------------------------------------------------------------------


async def _call_failing_tool(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> CallToolResult:
    """Call list_workspaces over the SDK after making the tool raise ``exc``.

    Swapped before register_tools runs, so the registration path under test (the
    one production uses) wraps the failing function.
    """

    async def list_workspaces(ctx: McpContext) -> list[dict[str, Any]]:
        raise exc

    monkeypatch.setattr(src.mcp_server.tools, "list_workspaces", list_workspaces)
    mcp = MCPServer("test")
    register_tools(mcp)
    async with Client(mcp) as client:
        return await client.call_tool("list_workspaces", {})


def _error_text(result: CallToolResult) -> str:
    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    return content.text


@pytest.mark.parametrize(
    "exc",
    [
        ForbiddenError("Not a member of this workspace"),
        NotFoundError("Research thread 'x' not found"),
        InputValidationError("Invalid workspace_id: 'nope'"),
        ServiceUnavailableError("Rate limit backend unavailable"),
        UnauthorizedError("Authentication required"),
        AppError(),
    ],
    ids=lambda exc: type(exc).__name__,
)
async def test_app_error_detail_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, exc: AppError
) -> None:
    # The same text REST shows the same user; the model needs it to self-correct.
    result = await _call_failing_tool(monkeypatch, exc)

    assert _error_text(result) == f"Error executing tool list_workspaces: {exc.detail}"


@pytest.mark.parametrize(
    ("exc", "logged"),
    [
        pytest.param(ServiceUnavailableError("down"), True, id="503"),
        pytest.param(AppError(), True, id="500"),
        pytest.param(ForbiddenError("Not a member"), False, id="403"),
        pytest.param(RateLimitError(retry_after=17), False, id="429"),
    ],
)
async def test_server_side_failures_are_logged_as_errors(
    monkeypatch: pytest.MonkeyPatch, exc: AppError, logged: bool
) -> None:
    # REST's rule: a 5xx is the server's fault and is logged at ERROR. The SDK
    # logs every tool error at INFO, which is enough for a caller's mistake.
    with structlog.testing.capture_logs() as logs:
        await _call_failing_tool(monkeypatch, exc)

    errors = [entry for entry in logs if entry["log_level"] == "error"]
    expected = {
        "event": "tool error",
        "log_level": "error",
        "tool": "list_workspaces",
        "error_type": type(exc).__name__,
        "status_code": exc.status_code,
    }
    assert errors == ([expected] if logged else [])


async def test_rate_limit_error_tells_the_model_when_to_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REST sends the wait in Retry-After; a model has only the text.
    result = await _call_failing_tool(monkeypatch, RateLimitError(retry_after=17))

    assert _error_text(result) == (
        "Error executing tool list_workspaces: "
        "Rate limit exceeded — retry in 17 seconds"
    )


async def test_unexpected_error_text_stays_on_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _call_failing_tool(
        monkeypatch, RuntimeError("connect to postgres://kp:s3cret@db failed")
    )

    assert _error_text(result) == "Error executing tool list_workspaces"


async def test_registration_keeps_every_tools_schema() -> None:
    # The SDK builds schemas and finds ctx from the signature; the error wrapper
    # must be invisible to it.
    wrapped, plain = MCPServer("wrapped"), MCPServer("plain")
    register_tools(wrapped)
    for fn in (
        list_workspaces,
        search_documents,
        list_documents,
        read_document,
        start_research,
        get_research_status,
    ):
        plain.add_tool(fn)

    async def schemas(mcp: MCPServer) -> dict[str, tuple[Any, Any]]:
        return {
            t.name: (t.input_schema, t.output_schema) for t in await mcp.list_tools()
        }

    assert await schemas(wrapped) == await schemas(plain)


# The tools that take a workspace_id, each with its other arguments and the
# service call that uses the workspace.
WORKSPACE_TOOLS: dict[str, tuple[dict[str, Any], str, str]] = {
    "search_documents": ({"query": "q"}, "search", "search"),
    "list_documents": ({}, "document", "list"),
    "read_document": ({"document_id": str(uuid.uuid4())}, "document", "read"),
    "start_research": ({"topic": "t"}, "research", "start_research"),
    "get_research_status": ({"thread_id": str(uuid.uuid4())}, "research", "get_status"),
}


@pytest.mark.parametrize("tool", WORKSPACE_TOOLS)
async def test_non_member_is_refused_before_the_workspace_is_used(
    tool: str, mock_user: User, fake_services: MagicMock, mock_redis: None
) -> None:
    # A workspace_id from the model is never trusted without a membership check,
    # and the refusal reaches the model through the registered tool's error path.
    args, service, method = WORKSPACE_TOOLS[tool]
    fake_services.workspace.get_user_role = AsyncMock(
        side_effect=ForbiddenError("Not a member of this workspace")
    )
    mcp = MCPServer("test")
    register_tools(mcp)

    with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
        async with Client(mcp) as client:
            result = await client.call_tool(
                tool, {"workspace_id": str(uuid.uuid4()), **args}
            )

    assert _error_text(result) == (
        f"Error executing tool {tool}: Not a member of this workspace"
    )
    getattr(getattr(fake_services, service), method).assert_not_called()


async def test_every_tool_is_covered_by_the_error_tests() -> None:
    # list_workspaces by the tests above that swap it, the rest by the
    # non-member test: a new tool must join one of them.
    mcp = MCPServer("test")
    register_tools(mcp)

    names = {tool.name for tool in await mcp.list_tools()}

    assert names == {"list_workspaces", *WORKSPACE_TOOLS}


# ADR-022: 2,000 characters after stripping, the same as REST.
_QUERY_CASES = [
    pytest.param("x" * 2000, "x" * 2000, id="at-cap"),
    pytest.param("  " + "x" * 2000 + "\n", "x" * 2000, id="stripped-then-counted"),
    pytest.param("x" * 2001, None, id="over-cap"),
    pytest.param(" \t\n", None, id="blank"),
]


@pytest.mark.parametrize(("query", "searched"), _QUERY_CASES)
async def test_search_query_is_capped_and_stripped(
    query: str,
    searched: str | None,
    mock_user: User,
    fake_services: MagicMock,
    mock_redis: None,
) -> None:
    fake_services.workspace.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
    fake_services.search.search = AsyncMock(return_value=_make_search_response(0))
    mcp = MCPServer("test")
    register_tools(mcp)
    ws_id = uuid.uuid4()

    with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "search_documents", {"workspace_id": str(ws_id), "query": query}
            )

    if searched is None:
        assert "query" in _error_text(result)
        fake_services.search.search.assert_not_called()
    else:
        assert not result.is_error
        fake_services.search.search.assert_awaited_once_with(
            workspace_id=ws_id, query=searched, top_k=5
        )


async def test_search_query_cap_is_in_the_schema() -> None:
    schema = (await _input_schemas())["search_documents"]["properties"]["query"]

    assert (schema["minLength"], schema["maxLength"]) == (1, 2000)


async def _input_schemas() -> dict[str, dict[str, Any]]:
    mcp = MCPServer("test")
    register_tools(mcp)
    return {tool.name: tool.input_schema for tool in await mcp.list_tools()}


@pytest.mark.parametrize("tool", WORKSPACE_TOOLS)
async def test_workspace_id_is_a_uuid_in_the_schema(tool: str) -> None:
    # The model sees the expected shape before it calls.
    schema = (await _input_schemas())[tool]

    assert schema["properties"]["workspace_id"]["format"] == "uuid"


@pytest.mark.parametrize("tool", WORKSPACE_TOOLS)
async def test_malformed_workspace_id_is_refused_before_the_tool_runs(
    tool: str, mock_user: User, fake_services: MagicMock
) -> None:
    # The SDK validates arguments against the schema first: a malformed id spends
    # no rate-limit budget and opens no unit of work, and the model learns which
    # argument to fix.
    args, _, _ = WORKSPACE_TOOLS[tool]
    redis = MagicMock()
    mcp = MCPServer("test")
    register_tools(mcp)

    with (
        patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user),
        patch("src.mcp_server.tools.get_async_redis_client", return_value=redis),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                tool, {**args, "workspace_id": "not-a-uuid"}
            )

    text = _error_text(result)
    assert "workspace_id" in text
    assert "valid UUID" in text
    redis.pipeline.assert_not_called()
    fake_services.workspace.get_user_role.assert_not_called()


async def test_thread_id_is_a_uuid_passed_on_in_canonical_form(
    mock_user: User, fake_services: MagicMock
) -> None:
    # Threads are stored under str(uuid4()); any spelling the model sends of the
    # same id must find that thread.
    thread_id = uuid.uuid4()
    fake_services.workspace.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
    fake_services.research.get_status = AsyncMock(
        return_value=ResearchStatusResponse(
            thread_id=str(thread_id),
            status="running",
            topic="t",
            plan=None,
            findings_count=0,
            synthesis=None,
        )
    )
    mcp = MCPServer("test")
    register_tools(mcp)

    with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_research_status",
                {
                    "workspace_id": str(uuid.uuid4()),
                    "thread_id": str(thread_id).upper(),
                },
            )

    assert not result.is_error, result.content
    assert (await _input_schemas())["get_research_status"]["properties"]["thread_id"][
        "format"
    ] == "uuid"
    fake_services.research.get_status.assert_awaited_once_with(
        workspace_id=ANY, user_id=mock_user.id, thread_id=str(thread_id)
    )


async def test_every_tool_declares_its_effects() -> None:
    # Clients base approval policy on these hints (e.g. Claude Code's plan mode runs
    # only read-only tools). Unset, the other hints default to destructive and
    # open-world, which no tool here is.
    mcp = MCPServer("test")
    register_tools(mcp)

    annotations = {tool.name: tool.annotations for tool in await mcp.list_tools()}

    read_only = {name for name, a in annotations.items() if a and a.read_only_hint}
    assert read_only == {
        "list_workspaces",
        "search_documents",
        "list_documents",
        "read_document",
        "get_research_status",
    }
    for name, a in annotations.items():
        assert a is not None, name
        assert a.read_only_hint is not None, name
        assert a.open_world_hint is False, name
        if not a.read_only_hint:
            assert a.destructive_hint is False, name


def _is_typed_object(schema: dict[str, Any] | None) -> bool:
    """An object schema with named fields — not untyped, not the SDK's wrapper."""
    if not schema or schema.get("additionalProperties") is True:
        return False
    properties = schema.get("properties", {})
    return bool(properties) and list(properties) != ["result"]


async def test_text_and_structured_content_carry_the_same_data(
    mock_user: User, fake_services: MagicMock
) -> None:
    # Clients differ in which channel reaches their model (Claude Code: structured
    # only), so both must say the same thing.
    fake_services.workspace.list_for_user = AsyncMock(
        return_value=[_workspace_info("Infra", "Runbooks")]
    )
    mcp = MCPServer("test")
    register_tools(mcp)

    with patch("src.mcp_server.tools.get_mcp_user", return_value=mock_user):
        async with Client(mcp) as client:
            result = await client.call_tool("list_workspaces", {})

    [block] = result.content
    assert isinstance(block, TextContent)
    assert json.loads(block.text) == result.structured_content


async def test_tools_publish_typed_output_schemas() -> None:
    # Whichever channel a client's model reads, the output model defines it.
    mcp = MCPServer("test")
    register_tools(mcp)

    schemas = {tool.name: tool.output_schema for tool in await mcp.list_tools()}

    typed = {name for name, schema in schemas.items() if _is_typed_object(schema)}
    assert typed == set(schemas) - {"start_research", "get_research_status"}


# ---------------------------------------------------------------------------
# Prompts: tool descriptions and server instructions
# ---------------------------------------------------------------------------

# Tool, argument and field names are snake_case; prose words are not.
_IDENTIFIER = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")

# What the instructions' loop is built from. The research tools stay out until
# Track 4 replaces them: a failed run reports "running" forever, so steering a
# model there invites endless polling. Track 4 must update this.
RETRIEVAL_TOOLS = {
    "list_workspaces",
    "search_documents",
    "list_documents",
    "read_document",
}


def _property_names(node: Any) -> set[str]:
    """Every property name in a JSON schema, $defs included."""
    if isinstance(node, list):
        return set().union(*(_property_names(n) for n in node))
    if not isinstance(node, dict):
        return set()
    names = set(node.get("properties", {}))
    return names.union(*(_property_names(v) for v in node.values()))


async def test_prompts_name_only_what_exists() -> None:
    # Descriptions and instructions steer the model by name; a renamed tool,
    # argument or output field must not leave them pointing at nothing.
    mcp = build_mcp_server()
    tools = await mcp.list_tools()
    known = {t.name for t in tools}.union(
        *(
            _property_names(t.input_schema) | _property_names(t.output_schema)
            for t in tools
        )
    )
    texts = {"instructions": mcp.instructions or ""}
    texts.update({t.name: t.description or "" for t in tools})

    unknown = {
        where: sorted(set(_IDENTIFIER.findall(text)) - known)
        for where, text in texts.items()
    }

    assert {where: names for where, names in unknown.items() if names} == {}


async def test_instructions_cover_the_retrieval_loop() -> None:
    mcp = build_mcp_server()
    registered = {t.name for t in await mcp.list_tools()}

    named = set(_IDENTIFIER.findall(mcp.instructions or ""))

    assert named & registered == RETRIEVAL_TOOLS
