"""Integration tests for the MCP server against real PostgreSQL.

Tools and auth run on real services built by the composition root over
the test DB session; only external calls (embeddings, rate limiting) are stubbed.
Protocol-level tests through the MCP client live in test_mcp_protocol.py.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import ExitStack, asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.composition import Services
from src.core.exceptions import ForbiddenError, InvalidTokenError
from src.domain.roles import WorkspaceRole
from src.domain.search import SearchResult, SearchResults
from src.mcp_server.auth import McpContext, _authenticate_request
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
        for module in ("auth", "tools"):
            stack.enter_context(patch(f"src.mcp_server.{module}.open_services", _open))
        stack.enter_context(
            patch("src.mcp_server.tools._mcp_rate_limit", new=AsyncMock())
        )
        yield services


@pytest.fixture
def as_user(mcp_user: User) -> McpContext:
    """The tool Context of a request MCP auth let through as ``mcp_user``."""
    scope = {"type": "http", "path": "/mcp", "state": {"mcp_user": mcp_user}}
    request_context = ServerRequestContext(
        request_id=1,
        protocol_version="2026-07-28",
        method="tools/call",
        meta=None,
        session=MagicMock(),
        lifespan_context=None,
        request=Request(scope),
    )
    return Context(request_context=request_context)


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

        with pytest.raises(InvalidTokenError, match="Invalid or revoked"):
            await _authenticate_request(_bearer_request(raw_key))


# ---------------------------------------------------------------------------
# Tools — real membership checks; only the embedding-backed search is stubbed
# ---------------------------------------------------------------------------


class TestSearchDocumentsIntegration:
    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, as_user: McpContext, mcp_workspace: uuid.UUID, real_services: Services
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
            as_user, workspace_id=str(mcp_workspace), query="test", top_k=5
        )

        search_svc.search.assert_awaited_once_with(
            workspace_id=mcp_workspace, query="test", top_k=5
        )
        assert len(result) == 1
        assert result[0]["chunk_text"] == "relevant chunk"
        assert result[0]["document_title"] == "Test Doc"

    @pytest.mark.asyncio
    async def test_search_rejects_non_member(
        self, as_user: McpContext, real_services: Services
    ) -> None:
        with pytest.raises(ForbiddenError):
            await search_documents(
                as_user, workspace_id=str(uuid.uuid4()), query="test"
            )
