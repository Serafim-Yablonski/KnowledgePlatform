"""Unit tests for MCP tools and auth helpers.

Services are mocked via Protocol stubs — no DB or Redis required.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.security.utils import get_authorization_scheme_param
from mcp import Client
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent
from starlette.requests import Request

import src.mcp_server.tools
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
from src.domain.roles import WorkspaceRole
from src.domain.workspace import WorkspaceInfo
from src.mcp_server.auth import McpContext, _authenticate_request, get_mcp_user
from src.mcp_server.tools import (
    ask_question,
    get_research_status,
    list_workspaces,
    register_tools,
    search_documents,
    start_research,
)
from src.models.user import User
from src.schemas.ai import AnswerResponse, SourceReference
from src.schemas.research import ResearchPlanResponse, ResearchStatusResponse
from src.schemas.search import SearchResponse, SearchResultItem

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


def _make_search_response(n: int = 1) -> SearchResponse:
    doc_id = uuid.uuid4()
    return SearchResponse(
        results=[
            SearchResultItem(
                chunk_text="relevant text",
                document_id=doc_id,
                document_title="Test Doc",
                score=0.9,
                chunk_metadata={},
            )
            for _ in range(n)
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


class TestListWorkspacesTool:
    async def test_returns_each_workspace_with_callers_role(
        self, fake_services: MagicMock
    ) -> None:
        user = _make_user()
        ws_id = uuid.uuid4()
        workspace = WorkspaceInfo(
            id=ws_id,
            name="Test Workspace",
            slug="test-workspace",
            description=None,
            is_active=True,
            created_at=datetime(2026, 1, 1),
            member_count=3,
        )
        fake_services.workspace.list_for_user = AsyncMock(return_value=[workspace])
        fake_services.workspace.get_user_role = AsyncMock(
            return_value=WorkspaceRole.VIEWER
        )

        result = await list_workspaces(ctx=_ctx(user))

        fake_services.workspace.list_for_user.assert_awaited_once_with(user)
        assert result == [
            {
                "id": str(ws_id),
                "name": "Test Workspace",
                "slug": "test-workspace",
                "role": "viewer",
                "member_count": 3,
            }
        ]


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
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_no_credentials_raises(self) -> None:
        request = MagicMock()
        request.headers = {}

        with pytest.raises(UnauthorizedError, match="Authentication required"):
            await _authenticate_request(request)

    @pytest.mark.asyncio
    async def test_whitespace_only_token_is_no_credentials(self) -> None:
        # Not InvalidTokenError: nothing was sent, so the challenge carries no error.
        request = MagicMock()
        request.headers = {"Authorization": "Bearer   "}

        with pytest.raises(UnauthorizedError) as exc_info:
            await _authenticate_request(request)
        assert exc_info.type is UnauthorizedError

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_invalid_workspace_id_raises(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        with pytest.raises(InputValidationError):
            await search_documents(
                _ctx(mock_user), workspace_id="not-a-uuid", query="q"
            )


# ---------------------------------------------------------------------------
# Tool: search_documents
# ---------------------------------------------------------------------------


class TestSearchDocumentsTool:
    @pytest.mark.asyncio
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
                _ctx(mock_user), workspace_id=str(ws_id), query="test query", top_k=3
            )

        mock_search_svc.search.assert_awaited_once_with(
            workspace_id=ws_id, query="test query", top_k=3
        )
        assert len(result) == 2
        assert result[0]["document_title"] == "Test Doc"
        assert "score" in result[0]

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated(self) -> None:
        with pytest.raises(UnauthorizedError):
            await search_documents(
                _ctx(None), workspace_id=str(uuid.uuid4()), query="q"
            )

    @pytest.mark.asyncio
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
                _ctx(mock_user), workspace_id=str(uuid.uuid4()), query="q"
            )

    @pytest.mark.asyncio
    async def test_formats_results_correctly(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        doc_id = uuid.uuid4()
        search_response = SearchResponse(
            results=[
                SearchResultItem(
                    chunk_text="chunk content",
                    document_id=doc_id,
                    document_title="My Document",
                    score=0.87654,
                    chunk_metadata={"page": 1},
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
                _ctx(mock_user), workspace_id=str(uuid.uuid4()), query="q"
            )

        assert result[0]["document_id"] == str(doc_id)
        assert result[0]["document_title"] == "My Document"
        assert result[0]["chunk_text"] == "chunk content"
        assert result[0]["score"] == 0.8765  # rounded to 4 dp
        assert result[0]["metadata"] == {"page": 1}


# ---------------------------------------------------------------------------
# Tool: ask_question
# ---------------------------------------------------------------------------


class TestAskQuestionTool:
    @pytest.mark.asyncio
    async def test_calls_ai_service_with_role(
        self, mock_user: User, fake_services: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        answer = AnswerResponse(
            answer="42",
            confidence=0.95,
            reasoning="Because.",
            sources=[
                SourceReference(
                    document_id=uuid.uuid4(),
                    document_title="Doc",
                    chunk_text="text",
                    relevance_score=0.9,
                )
            ],
        )

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_ai_svc = MagicMock()
        mock_ai_svc.ask = AsyncMock(return_value=answer)

        with (
            patch.object(fake_services, "workspace", mock_workspace_svc),
            patch.object(fake_services, "ai", mock_ai_svc),
        ):
            result = await ask_question(
                _ctx(mock_user), workspace_id=str(ws_id), question="What is the answer?"
            )

        mock_ai_svc.ask.assert_awaited_once_with(
            workspace_id=ws_id,
            user_id=mock_user.id,
            question="What is the answer?",
            role=WorkspaceRole.MEMBER,
        )
        assert result["answer"] == "42"
        assert result["confidence"] == 0.95
        assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated(self) -> None:
        with pytest.raises(UnauthorizedError):
            await ask_question(_ctx(None), workspace_id=str(uuid.uuid4()), question="q")


# ---------------------------------------------------------------------------
# Tool: start_research
# ---------------------------------------------------------------------------


class TestStartResearchTool:
    @pytest.mark.asyncio
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
                workspace_id=str(ws_id),
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
    @pytest.mark.asyncio
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
                _ctx(mock_user), thread_id=thread_id, workspace_id=str(ws_id)
            )

        assert result["thread_id"] == thread_id
        assert result["status"] == "completed"
        assert result["synthesis"] == "The answer is ..."
        assert result["plan"]["queries"] == ["q1"]
        assert result["human_approved"] is True

    @pytest.mark.asyncio
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
                _ctx(mock_user), thread_id=thread_id, workspace_id=str(ws_id)
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
        ask_question,
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
    "ask_question": ({"question": "q"}, "ai", "ask"),
    "start_research": ({"topic": "t"}, "research", "start_research"),
    "get_research_status": ({"thread_id": "t"}, "research", "get_status"),
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
