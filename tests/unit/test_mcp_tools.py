"""Unit tests for MCP tools and auth helpers.

Services are mocked via Protocol stubs — no DB or Redis required.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ForbiddenError, InputValidationError, UnauthorizedError
from src.domain.roles import WorkspaceRole
from src.mcp_server.auth import _authenticate_request, _current_user, get_mcp_user
from src.mcp_server.tools import (
    ask_question,
    get_research_status,
    search_documents,
    start_research,
)
from src.schemas.ai import AnswerResponse, SourceReference
from src.schemas.research import ResearchPlanResponse, ResearchStatusResponse
from src.schemas.search import SearchResponse, SearchResultItem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.is_active = True
    return user


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
def mock_user():
    user = _make_user()
    token = _current_user.set(user)
    yield user
    _current_user.reset(token)


@pytest.fixture
def mock_session_ctx():
    """Patch get_session to return a no-op context manager (no DB access)."""
    session = MagicMock()

    @asynccontextmanager
    async def _fake_get_session() -> AsyncGenerator[MagicMock]:
        yield session

    with patch("src.mcp_server.tools.get_session", _fake_get_session):
        yield session


@pytest.fixture
def mock_redis():
    with patch("src.mcp_server.tools.get_async_redis_client", return_value=MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestGetMcpUser:
    def test_returns_user_when_set(self) -> None:
        user = _make_user()
        token = _current_user.set(user)
        try:
            assert get_mcp_user() is user
        finally:
            _current_user.reset(token)

    def test_raises_when_not_set(self) -> None:
        token = _current_user.set(None)
        try:
            with pytest.raises(UnauthorizedError):
                get_mcp_user()
        finally:
            _current_user.reset(token)


class TestAuthenticateRequest:
    @pytest.mark.asyncio
    async def test_bearer_token_valid(self) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer valid-token"}
        expected_user = _make_user()

        with (
            patch("src.mcp_server.auth.get_session") as mock_gs,
            patch("src.mcp_server.auth.SQLAlchemyUserRepository"),
            patch("src.mcp_server.auth.AuthService") as mock_auth_svc_cls,
        ):
            session = MagicMock()

            @asynccontextmanager
            async def _fake() -> AsyncGenerator[MagicMock]:
                yield session

            mock_gs.return_value = _fake()
            mock_auth_svc_cls.return_value.get_current_user = AsyncMock(
                return_value=expected_user
            )

            result = await _authenticate_request(request)

        assert result is expected_user

    @pytest.mark.asyncio
    async def test_no_credentials_raises(self) -> None:
        request = MagicMock()
        request.headers = {}

        with pytest.raises(UnauthorizedError, match="Authentication required"):
            await _authenticate_request(request)

    @pytest.mark.asyncio
    async def test_api_key_no_user_email_configured_raises(self) -> None:
        request = MagicMock()
        request.headers = {"X-API-Key": "my-key"}

        with patch("src.mcp_server.auth.settings") as mock_settings:
            mock_settings.MCP_API_KEY = "my-key"
            mock_settings.MCP_API_KEY_USER_EMAIL = None

            with pytest.raises(ForbiddenError, match="email not configured"):
                await _authenticate_request(request)

    @pytest.mark.asyncio
    async def test_wrong_api_key_raises(self) -> None:
        request = MagicMock()
        request.headers = {"X-API-Key": "wrong-key"}

        with patch("src.mcp_server.auth.settings") as mock_settings:
            mock_settings.MCP_API_KEY = "correct-key"
            mock_settings.MCP_API_KEY_USER_EMAIL = "admin@example.com"

            with pytest.raises(UnauthorizedError):
                await _authenticate_request(request)

    @pytest.mark.asyncio
    async def test_invalid_workspace_id_raises(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
    ) -> None:
        with pytest.raises(InputValidationError):
            await search_documents(workspace_id="not-a-uuid", query="q")


# ---------------------------------------------------------------------------
# Tool: search_documents
# ---------------------------------------------------------------------------


class TestSearchDocumentsTool:
    @pytest.mark.asyncio
    async def test_calls_search_service(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        search_response = _make_search_response(2)

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_search_svc = MagicMock()
        mock_search_svc.search = AsyncMock(return_value=search_response)

        with (
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_search_service",
                return_value=mock_search_svc,
            ),
        ):
            result = await search_documents(
                workspace_id=str(ws_id), query="test query", top_k=3
            )

        mock_search_svc.search.assert_awaited_once_with(
            workspace_id=ws_id, query="test query", top_k=3
        )
        assert len(result) == 2
        assert result[0]["document_title"] == "Test Doc"
        assert "score" in result[0]

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated(self) -> None:
        token = _current_user.set(None)
        try:
            with pytest.raises(UnauthorizedError):
                await search_documents(workspace_id=str(uuid.uuid4()), query="q")
        finally:
            _current_user.reset(token)

    @pytest.mark.asyncio
    async def test_rejects_non_member(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
    ) -> None:
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(
            side_effect=ForbiddenError("Not a member")
        )

        with (
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            pytest.raises(ForbiddenError),
        ):
            await search_documents(workspace_id=str(uuid.uuid4()), query="q")

    @pytest.mark.asyncio
    async def test_formats_results_correctly(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
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
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_search_service",
                return_value=mock_search_svc,
            ),
        ):
            result = await search_documents(workspace_id=str(uuid.uuid4()), query="q")

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
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
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
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_ai_service",
                return_value=mock_ai_svc,
            ),
        ):
            result = await ask_question(
                workspace_id=str(ws_id), question="What is the answer?"
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
        token = _current_user.set(None)
        try:
            with pytest.raises(UnauthorizedError):
                await ask_question(workspace_id=str(uuid.uuid4()), question="q")
        finally:
            _current_user.reset(token)


# ---------------------------------------------------------------------------
# Tool: start_research
# ---------------------------------------------------------------------------


class TestStartResearchTool:
    @pytest.mark.asyncio
    async def test_calls_research_service(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
    ) -> None:
        ws_id = uuid.uuid4()
        thread_id = str(uuid.uuid4())

        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_research_svc = MagicMock()
        mock_research_svc.start_research = AsyncMock(return_value=thread_id)

        with (
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_research_service",
                return_value=mock_research_svc,
            ),
        ):
            result = await start_research(
                workspace_id=str(ws_id), topic="AI trends", max_iterations=2
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
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
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
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_research_service",
                return_value=mock_research_svc,
            ),
        ):
            result = await get_research_status(
                thread_id=thread_id, workspace_id=str(ws_id)
            )

        assert result["thread_id"] == thread_id
        assert result["status"] == "completed"
        assert result["synthesis"] == "The answer is ..."
        assert result["plan"]["queries"] == ["q1"]
        assert result["human_approved"] is True

    @pytest.mark.asyncio
    async def test_omits_plan_when_none(
        self, mock_user: MagicMock, mock_session_ctx: MagicMock, mock_redis: None
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
            patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ),
            patch(
                "src.mcp_server.tools._make_research_service",
                return_value=mock_research_svc,
            ),
        ):
            result = await get_research_status(
                thread_id=thread_id, workspace_id=str(ws_id)
            )

        assert "plan" not in result
        assert result["synthesis"] is None
