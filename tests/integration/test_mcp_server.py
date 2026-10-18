"""Integration tests for the MCP server against real PostgreSQL.

Tools, resources and auth run on real services built by the composition root over
the test DB session; only external calls (embeddings, rate limiting) are stubbed.
Protocol-level tests through the MCP client arrive with the transport rewrite.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import ExitStack, asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.composition import Services
from src.core.exceptions import ForbiddenError
from src.core.security import create_access_token
from src.domain.documents import ContentType
from src.domain.roles import WorkspaceRole
from src.domain.search import SearchResult, SearchResults
from src.mcp_server.auth import _authenticate_request, _current_user
from src.mcp_server.resources import get_workspace_stats, list_workspace_documents
from src.mcp_server.tools import search_documents
from src.models.user import User
from src.repositories.document import SQLAlchemyDocumentRepository
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


@pytest.fixture
async def services(db_session: AsyncSession) -> AsyncGenerator[Services]:
    fake_redis = FakeRedis()
    yield Services(db_session, fake_redis, http_client=MagicMock())
    await fake_redis.aclose()


@pytest.fixture
def real_services(services: Services) -> Generator[Services]:
    """Route every MCP entry point's open_services() to the test-DB services."""

    @asynccontextmanager
    async def _open() -> AsyncGenerator[Services]:
        yield services

    with ExitStack() as stack:
        for module in ("auth", "tools", "resources"):
            stack.enter_context(patch(f"src.mcp_server.{module}.open_services", _open))
        stack.enter_context(
            patch("src.mcp_server.tools._mcp_rate_limit", new=AsyncMock())
        )
        yield services


@pytest.fixture
def as_user(mcp_user: User) -> Generator[User]:
    token = _current_user.set(mcp_user)
    yield mcp_user
    _current_user.reset(token)


def _bearer_request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


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
    async def test_unknown_api_key_returns_403(
        self, async_client: AsyncClient, real_services: Services
    ) -> None:
        response = await async_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_bearer_token_passes_auth_then_hits_host_check(
        self, async_client: AsyncClient, bearer_token: str, real_services: Services
    ) -> None:
        from src.mcp_server.server import get_mcp_session_manager

        # ASGITransport skips the lifespan, which normally runs the session manager.
        async with get_mcp_session_manager().run():
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
        # Auth passed and the request reached the transport, which rejects the
        # non-localhost Host (plan finding F2). Track 2 fixes the allowlist and
        # replaces this with a full initialize → tools/call test.
        assert response.status_code == 421


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_db_api_key_authenticates_and_records_use(
        self, mcp_user: User, db_session: AsyncSession, real_services: Services
    ) -> None:
        api_key, raw_key = await real_services.api_key.create(mcp_user.id, "laptop")
        assert api_key.last_used_at is None

        user = await _authenticate_request(_bearer_request(raw_key))

        assert user.id == mcp_user.id
        await db_session.refresh(api_key)
        assert api_key.last_used_at is not None

    @pytest.mark.asyncio
    async def test_failed_use_write_does_not_break_auth_on_cache_miss(
        self,
        mcp_user: User,
        db_session: AsyncSession,
        real_services: Services,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression (ADR-019): on a cache miss the key is loaded into the session;
        # the failed UPDATE's rollback must not expire the user auth returns.
        _, raw_key = await real_services.api_key.create(mcp_user.id, "laptop")
        user_id = mcp_user.id
        db_session.expunge_all()  # like production: a fresh session per auth

        real_execute = db_session.execute

        async def _fail_updates(statement: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(statement, sa.Update):
                raise OperationalError("UPDATE api_keys", {}, Exception("DB hiccup"))
            return await real_execute(statement, *args, **kwargs)

        monkeypatch.setattr(db_session, "execute", _fail_updates)

        user = await _authenticate_request(_bearer_request(raw_key))

        assert user.id == user_id

    @pytest.mark.asyncio
    async def test_deactivated_api_key_is_rejected(
        self, mcp_user: User, real_services: Services
    ) -> None:
        api_key, raw_key = await real_services.api_key.create(mcp_user.id, "old")
        await real_services.api_key.deactivate(api_key.id, mcp_user.id)

        with pytest.raises(ForbiddenError, match="Invalid or revoked"):
            await _authenticate_request(_bearer_request(raw_key))


# ---------------------------------------------------------------------------
# Tools — real membership checks; only the embedding-backed search is stubbed
# ---------------------------------------------------------------------------


class TestSearchDocumentsIntegration:
    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, as_user: User, mcp_workspace: uuid.UUID, real_services: Services
    ) -> None:
        search_svc = MagicMock()
        search_svc.search = AsyncMock(
            return_value=SearchResults(
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
        )
        real_services.search = search_svc  # type: ignore[misc]

        result = await search_documents(
            workspace_id=str(mcp_workspace), query="test", top_k=5
        )

        search_svc.search.assert_awaited_once_with(
            workspace_id=mcp_workspace, query="test", top_k=5
        )
        assert len(result) == 1
        assert result[0]["chunk_text"] == "relevant chunk"
        assert result[0]["document_title"] == "Test Doc"

    @pytest.mark.asyncio
    async def test_search_rejects_non_member(
        self, as_user: User, real_services: Services
    ) -> None:
        with pytest.raises(ForbiddenError):
            await search_documents(workspace_id=str(uuid.uuid4()), query="test")


# ---------------------------------------------------------------------------
# Resources — fully real
# ---------------------------------------------------------------------------


class TestWorkspaceResources:
    @pytest.fixture
    async def document_id(
        self, db_session: AsyncSession, mcp_user: User, mcp_workspace: uuid.UUID
    ) -> uuid.UUID:
        doc = await SQLAlchemyDocumentRepository(db_session).create(
            workspace_id=mcp_workspace,
            title="Sample Doc",
            content_type=ContentType.PLAINTEXT,
            file_path="/tmp/sample.txt",
            file_size_bytes=1024,
            uploaded_by=mcp_user.id,
        )
        return doc.id

    @pytest.mark.asyncio
    async def test_documents_resource_lists_workspace_documents(
        self,
        as_user: User,
        mcp_workspace: uuid.UUID,
        document_id: uuid.UUID,
        real_services: Services,
    ) -> None:
        docs = await list_workspace_documents(str(mcp_workspace))

        assert [d["id"] for d in docs] == [str(document_id)]
        assert docs[0]["title"] == "Sample Doc"
        assert "created_at" in docs[0]

    @pytest.mark.asyncio
    async def test_stats_resource_counts_documents(
        self,
        as_user: User,
        mcp_workspace: uuid.UUID,
        document_id: uuid.UUID,
        real_services: Services,
    ) -> None:
        data = await get_workspace_stats(str(mcp_workspace))

        assert data["document_count"] == 1
        assert data["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_resources_reject_non_member(
        self, as_user: User, real_services: Services
    ) -> None:
        with pytest.raises(ForbiddenError):
            await list_workspace_documents(str(uuid.uuid4()))
