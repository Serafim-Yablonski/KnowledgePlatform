"""Integration tests for the MCP server.

Tests cover:
- Auth middleware (HTTP level) — no MCP protocol needed
- Full tool calls via the MCP endpoint with bearer-token auth
- Resource access with workspace membership enforcement

The MCP server is mounted at /mcp in the main FastAPI app, so we reuse the
existing `async_client` fixture (httpx + ASGITransport, real PostgreSQL).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import create_access_token
from src.domain.roles import WorkspaceRole
from src.domain.search import SearchResult, SearchResults
from src.domain.workspace import WorkspaceStats
from src.mcp_server.auth import _current_user
from src.mcp_server.resources import get_workspace_stats, list_workspace_documents
from src.mcp_server.tools import search_documents
from src.models.user import User
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.auth import UserCreate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    return await repo.create(
        UserCreate(email="mcp-test@example.com", password="password123"),
        hashed_password="$2b$12$fixturehash",
    )


@pytest.fixture
async def mcp_workspace(db_session: AsyncSession, mcp_user: User) -> uuid.UUID:
    repo = SQLAlchemyWorkspaceRepository(db_session)
    ws = await repo.create(
        name="MCP Test Workspace",
        slug=f"mcp-test-{uuid.uuid4().hex[:8]}",
        created_by_id=mcp_user.id,
    )
    await repo.add_member(
        workspace_id=ws.id,
        user_id=mcp_user.id,
        role=WorkspaceRole.OWNER,
    )
    return ws.id


@pytest.fixture
def bearer_token(mcp_user: User) -> str:
    return create_access_token(mcp_user.id)


# ---------------------------------------------------------------------------
# Auth middleware (HTTP level)
# ---------------------------------------------------------------------------


class TestMCPAuthMiddleware:
    @pytest.mark.asyncio
    async def test_no_auth_header_returns_401(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == 401

    @pytest.mark.asyncio
    async def test_invalid_bearer_token_returns_403(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_bearer_token_passes_auth(
        self, async_client: AsyncClient, bearer_token: str
    ) -> None:
        response = await async_client.post(
            "/mcp",
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                }
            ),
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            },
        )
        # Auth passed — MCP protocol response (not a 403)
        assert response.status_code != 403

    @pytest.mark.asyncio
    async def test_api_key_auth(self, mcp_user: User, db_session: AsyncSession) -> None:
        from contextlib import asynccontextmanager

        from starlette.requests import Request

        from src.mcp_server.auth import _authenticate_request

        @asynccontextmanager
        async def _test_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        scope: dict = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "query_string": b"",
            "headers": [(b"x-api-key", b"test-api-key")],
        }
        request = Request(scope)

        with (
            patch("src.mcp_server.auth.settings") as mock_settings,
            patch(
                "src.mcp_server.auth.get_session", side_effect=lambda: _test_session()
            ),
        ):
            mock_settings.MCP_API_KEY = "test-api-key"
            mock_settings.MCP_API_KEY_USER_EMAIL = mcp_user.email

            user = await _authenticate_request(request)

        assert user.email == mcp_user.email


# ---------------------------------------------------------------------------
# Tool integration — call tool functions directly with real DB + ContextVar
# ---------------------------------------------------------------------------


class TestSearchDocumentsIntegration:
    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        mcp_user: User,
        mcp_workspace: uuid.UUID,
        db_session: AsyncSession,
    ) -> None:
        search_response = SearchResults(
            results=[
                SearchResult(
                    chunk_id=uuid.uuid4(),
                    chunk_text="relevant chunk",
                    document_id=uuid.uuid4(),
                    document_title="Test Doc",
                    score=0.88,
                    chunk_metadata={},
                )
            ],
            query="test",
            total_results=1,
        )

        token = _current_user.set(mcp_user)
        try:
            with (
                patch(
                    "src.mcp_server.tools._make_search_service"
                ) as mock_search_factory,
                patch(
                    "src.mcp_server.tools._make_workspace_service"
                ) as mock_ws_factory,
                patch("src.mcp_server.tools._mcp_rate_limit", new=AsyncMock()),
                patch("src.mcp_server.tools.get_session") as mock_gs,
                patch(
                    "src.mcp_server.tools.get_async_redis_client",
                    return_value=MagicMock(),
                ),
            ):
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def _fake_session() -> AsyncGenerator[MagicMock]:
                    yield MagicMock()

                mock_gs.side_effect = lambda: _fake_session()

                mock_ws_svc = MagicMock()
                mock_ws_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
                mock_ws_factory.return_value = mock_ws_svc

                mock_search_svc = MagicMock()
                mock_search_svc.search = AsyncMock(return_value=search_response)
                mock_search_factory.return_value = mock_search_svc

                result = await search_documents(
                    workspace_id=str(mcp_workspace),
                    query="test",
                    top_k=5,
                )
        finally:
            _current_user.reset(token)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["chunk_text"] == "relevant chunk"
        assert result[0]["document_title"] == "Test Doc"

    @pytest.mark.asyncio
    async def test_search_rejects_non_member(
        self,
        mcp_user: User,
        db_session: AsyncSession,
    ) -> None:
        other_workspace_id = uuid.uuid4()

        token = _current_user.set(mcp_user)
        try:
            with (
                patch(
                    "src.mcp_server.tools._make_workspace_service"
                ) as mock_ws_factory,
                patch("src.mcp_server.tools._mcp_rate_limit", new=AsyncMock()),
                patch("src.mcp_server.tools.get_session") as mock_gs,
                patch(
                    "src.mcp_server.tools.get_async_redis_client",
                    return_value=MagicMock(),
                ),
            ):
                from contextlib import asynccontextmanager

                from src.core.exceptions import ForbiddenError

                @asynccontextmanager
                async def _fake_session() -> AsyncGenerator[MagicMock]:
                    yield MagicMock()

                mock_gs.side_effect = lambda: _fake_session()

                mock_ws_svc = MagicMock()
                mock_ws_svc.get_user_role = AsyncMock(
                    side_effect=ForbiddenError("Not a member")
                )
                mock_ws_factory.return_value = mock_ws_svc

                with pytest.raises(ForbiddenError):
                    await search_documents(
                        workspace_id=str(other_workspace_id),
                        query="test",
                    )
        finally:
            _current_user.reset(token)


# ---------------------------------------------------------------------------
# Resource integration
# ---------------------------------------------------------------------------


class TestWorkspaceResources:
    @pytest.mark.asyncio
    async def test_documents_resource_returns_json(
        self,
        mcp_user: User,
        mcp_workspace: uuid.UUID,
    ) -> None:
        token = _current_user.set(mcp_user)
        try:
            with (
                patch(
                    "src.mcp_server.resources._make_workspace_service"
                ) as mock_ws_factory,
                patch(
                    "src.mcp_server.resources._make_document_service"
                ) as mock_doc_factory,
                patch("src.mcp_server.resources.get_session") as mock_gs,
            ):
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def _fake_session() -> AsyncGenerator[MagicMock]:
                    yield MagicMock()

                mock_gs.side_effect = lambda: _fake_session()

                mock_ws_svc = MagicMock()
                mock_ws_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.OWNER)
                mock_ws_factory.return_value = mock_ws_svc

                import datetime

                mock_doc = MagicMock()
                mock_doc.id = uuid.uuid4()
                mock_doc.title = "Sample Doc"
                mock_doc.content_type = "text/plain"
                mock_doc.status = "ready"
                mock_doc.version = 1
                mock_doc.file_size_bytes = 1024
                mock_doc.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)

                mock_doc_svc = MagicMock()
                mock_doc_svc._list_by_workspace_id = AsyncMock(return_value=[mock_doc])
                mock_doc_factory.return_value = mock_doc_svc

                raw = await list_workspace_documents(str(mcp_workspace))
        finally:
            _current_user.reset(token)

        docs = raw
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert docs[0]["title"] == "Sample Doc"
        assert "created_at" in docs[0]

    @pytest.mark.asyncio
    async def test_stats_resource_returns_json(
        self,
        mcp_user: User,
        mcp_workspace: uuid.UUID,
    ) -> None:
        token = _current_user.set(mcp_user)
        try:
            with (
                patch(
                    "src.mcp_server.resources._make_workspace_service"
                ) as mock_ws_factory,
                patch(
                    "src.mcp_server.resources._make_document_service"
                ) as mock_doc_factory,
                patch("src.mcp_server.resources.get_session") as mock_gs,
            ):
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def _fake_session() -> AsyncGenerator[MagicMock]:
                    yield MagicMock()

                mock_gs.side_effect = lambda: _fake_session()

                mock_ws_svc = MagicMock()
                mock_ws_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.OWNER)
                mock_ws_factory.return_value = mock_ws_svc

                stats = WorkspaceStats(
                    document_count=3,
                    chunk_count=42,
                    total_tokens_indexed=15000,
                    last_document_updated_at=None,
                )
                mock_doc_svc = MagicMock()
                mock_doc_svc._get_workspace_stats = AsyncMock(return_value=stats)
                mock_doc_factory.return_value = mock_doc_svc

                raw = await get_workspace_stats(str(mcp_workspace))
        finally:
            _current_user.reset(token)

        data = raw
        assert data["document_count"] == 3
        assert data["chunk_count"] == 42
        assert data["total_tokens_indexed"] == 15000
        assert data["last_document_updated_at"] is None
