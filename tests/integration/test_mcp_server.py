"""Integration tests for the MCP server against real PostgreSQL.

Tools and auth run on real services built by the composition root over
the test DB session; only external calls are faked: Gemini's HTTP API (the real
EmbeddingService runs) and rate limiting.
Protocol-level tests through the MCP client live in test_mcp_protocol.py.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.composition import Services
from src.core.exceptions import ForbiddenError, InvalidTokenError, NotFoundError
from src.domain.documents import ContentType, DocumentStatus
from src.domain.roles import WorkspaceRole
from src.mcp_server.auth import McpContext, _authenticate_request
from src.mcp_server.schemas import (
    ChunkContent,
    DocumentSummary,
    SearchHit,
    SearchHits,
)
from src.mcp_server.tools import (
    list_documents,
    list_workspaces,
    read_document,
    search_documents,
)
from src.models.chunk import EMBEDDING_DIMS, DocumentChunk
from src.models.document import Document
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


_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)


@pytest.fixture
def embedded() -> list[str]:
    """Texts the fake Gemini API was asked to embed."""
    return []


@pytest.fixture
async def services(
    db_session: AsyncSession, embedded: list[str]
) -> AsyncGenerator[Services]:
    def gemini(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        embedded.extend(r["content"]["parts"][0]["text"] for r in body["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": _VECTOR}]})

    fake_redis = FakeRedis()
    async with httpx.AsyncClient(transport=httpx.MockTransport(gemini)) as http:
        yield Services(db_session, fake_redis, http_client=http)
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
# Tools — real membership checks and real queries
# ---------------------------------------------------------------------------


async def _workspace_of(session: AsyncSession, user: User, name: str) -> uuid.UUID:
    repo = SQLAlchemyWorkspaceRepository(session)
    ws = await repo.create(
        name=name, slug=f"{name.lower()}-{uuid.uuid4().hex[:8]}", created_by_id=user.id
    )
    await repo.add_member(workspace_id=ws.id, user_id=user.id, role=WorkspaceRole.OWNER)
    return ws.id


async def _indexed_doc(
    session: AsyncSession, workspace_id: uuid.UUID, user: User, *, version: int
) -> Document:
    doc = Document(
        workspace_id=workspace_id,
        title=f"Doc v{version}",
        content_type=ContentType.MARKDOWN,
        file_path=f"/tmp/{uuid.uuid4()}.md",
        file_size_bytes=100,
        uploaded_by=user.id,
        status=DocumentStatus.INDEXED,
        version=version,
    )
    session.add(doc)
    await session.flush()
    return doc


async def _chunk(
    session: AsyncSession,
    doc: Document,
    *,
    version: int,
    index: int,
    section: str | None,
) -> None:
    session.add(
        DocumentChunk(
            document_id=doc.id,
            chunk_index=index,
            text=f"chunk {index} of v{version}",
            embedding=_VECTOR,
            token_count=5,
            metadata_={} if section is None else {"section": section},
            version=version,
        )
    )
    await session.flush()


async def _listed_doc(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user: User,
    *,
    created_at: datetime,
    status: DocumentStatus,
) -> Document:
    doc = Document(
        workspace_id=workspace_id,
        title=f"{status} at {created_at:%H:%M:%S.%f}",
        content_type=ContentType.PLAINTEXT,
        file_path=f"/tmp/{uuid.uuid4()}.txt",
        file_size_bytes=100,
        uploaded_by=user.id,
        status=status,
        version=1,
        created_at=created_at,
    )
    session.add(doc)
    await session.flush()
    return doc


class TestSearchDocumentsIntegration:
    async def test_search_runs_the_real_query(
        self,
        db_session: AsyncSession,
        mcp_user: User,
        as_user: McpContext,
        mcp_workspace: uuid.UUID,
        real_services: Services,
        embedded: list[str],
    ) -> None:
        # Every chunk below has the query's own vector, so only the SQL's filters
        # can tell them apart: a stale version, and another workspace the user is
        # also a member of (filtered by the query, not by membership).
        other = await _workspace_of(db_session, mcp_user, "Other")
        doc = await _indexed_doc(db_session, mcp_workspace, mcp_user, version=2)
        foreign = await _indexed_doc(db_session, other, mcp_user, version=1)
        await _chunk(db_session, doc, version=1, index=0, section="Old")
        await _chunk(db_session, doc, version=2, index=4, section="Setup > Install")
        await _chunk(db_session, foreign, version=1, index=0, section=None)

        result = await search_documents(
            as_user, workspace_id=mcp_workspace, query="how to install", top_k=5
        )

        assert embedded == ["how to install"]
        assert result == SearchHits(
            results=[
                SearchHit(
                    document_id=doc.id,
                    document_title="Doc v2",
                    chunk_index=4,
                    section="Setup > Install",
                    chunk_text="chunk 4 of v2",
                    score=1.0,
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_search_rejects_non_member(
        self, as_user: McpContext, real_services: Services
    ) -> None:
        with pytest.raises(ForbiddenError):
            await search_documents(as_user, workspace_id=uuid.uuid4(), query="test")


class TestListWorkspacesIntegration:
    async def test_one_query_and_no_lookups_per_workspace(
        self,
        db_session: AsyncSession,
        mcp_user: User,
        as_user: McpContext,
        real_services: Services,
    ) -> None:
        # A lookup per workspace multiplied the cost by the number of workspaces,
        # and failed the whole listing if one membership vanished in between.
        repo = SQLAlchemyWorkspaceRepository(db_session)
        created = []
        for i, description in enumerate(["Runbooks and postmortems", None, "ML"]):
            ws = await repo.create(
                name=f"ws-{i}",
                slug=f"ws-{i}-{uuid.uuid4().hex[:8]}",
                created_by_id=mcp_user.id,
                description=description,
            )
            await repo.add_member(ws.id, mcp_user.id, WorkspaceRole.VIEWER)
            created.append(ws)

        statements: list[str] = []
        engine = db_session.bind.sync_engine  # type: ignore[union-attr]

        def _record(*args: Any) -> None:
            statements.append(args[2])

        sa.event.listen(engine, "before_cursor_execute", _record)
        try:
            result = await list_workspaces(as_user)
        finally:
            sa.event.remove(engine, "before_cursor_execute", _record)

        # The cache starts empty, so a lookup per workspace would reach the DB.
        assert len(statements) == 1
        assert {(w.id, w.name, w.description) for w in result.workspaces} == {
            (ws.id, ws.name, ws.description) for ws in created
        }


class TestListDocumentsIntegration:
    async def test_pages_through_the_real_listing(
        self,
        db_session: AsyncSession,
        mcp_user: User,
        as_user: McpContext,
        mcp_workspace: uuid.UUID,
        real_services: Services,
    ) -> None:
        # Pages of 2 over t3 | t2 t2 | t1 t1: the t2 tie straddles the first page
        # boundary, so only the cursor's id tie-breaker keeps both. The newest
        # document sits in another workspace the user also belongs to: only the
        # query's filter keeps it out. Every status is listed, unlike search.
        t1, t2, t3 = (datetime(2026, 10, 23, 9, 0, s, 250_000) for s in (1, 2, 3))
        other = await _workspace_of(db_session, mcp_user, "Other")
        await _listed_doc(
            db_session,
            other,
            mcp_user,
            created_at=datetime(2026, 10, 23, 10, 0),
            status=DocumentStatus.INDEXED,
        )
        seeded = [
            await _listed_doc(
                db_session, mcp_workspace, mcp_user, created_at=at, status=status
            )
            for at, status in [
                (t1, DocumentStatus.INDEXED),
                (t1, DocumentStatus.FAILED),
                (t2, DocumentStatus.PENDING),
                (t2, DocumentStatus.INDEXED),
                (t3, DocumentStatus.PROCESSING),
            ]
        ]

        listed: list[DocumentSummary] = []
        page_sizes: list[int] = []
        cursor: str | None = None
        for _ in range(10):  # bounded: a cursor that never advances fails, not hangs
            page = await list_documents(
                as_user, workspace_id=mcp_workspace, cursor=cursor, limit=2
            )
            listed.extend(page.documents)
            page_sizes.append(len(page.documents))
            cursor = page.next_cursor
            if cursor is None:
                break

        newest_first = sorted(seeded, key=lambda d: (d.created_at, d.id), reverse=True)
        assert page_sizes == [2, 2, 1]
        assert listed == [
            DocumentSummary(
                id=d.id,
                title=d.title,
                content_type=d.content_type,
                status=d.status,
                created_at=d.created_at,
            )
            for d in newest_first
        ]


class TestReadDocumentIntegration:
    async def test_reads_the_current_version_to_the_end(
        self,
        db_session: AsyncSession,
        mcp_user: User,
        as_user: McpContext,
        mcp_workspace: uuid.UUID,
        real_services: Services,
    ) -> None:
        # Version 1's chunks are stale; pages of 2 over 5 chunks end with a short one.
        doc = await _indexed_doc(db_session, mcp_workspace, mcp_user, version=2)
        for i in range(7):
            await _chunk(db_session, doc, version=1, index=i, section=None)
        for i in (4, 0, 3, 1, 2):
            await _chunk(db_session, doc, version=2, index=i, section="## Ops")

        read: list[ChunkContent] = []
        page_sizes: list[int] = []
        totals: set[int] = set()
        start: int | None = 0
        for _ in range(10):  # bounded: a start that never advances fails, not hangs
            assert start is not None
            page = await read_document(
                as_user,
                workspace_id=mcp_workspace,
                document_id=doc.id,
                start_chunk=start,
                max_chunks=2,
            )
            read.extend(page.chunks)
            page_sizes.append(len(page.chunks))
            totals.add(page.total_chunks)
            start = page.next_start_chunk
            if start is None:
                break

        assert page_sizes == [2, 2, 1]
        assert totals == {5}
        assert read == [
            ChunkContent(chunk_index=i, section="## Ops", chunk_text=f"chunk {i} of v2")
            for i in range(5)
        ]

    async def test_document_of_another_workspace_is_not_found(
        self,
        db_session: AsyncSession,
        mcp_user: User,
        as_user: McpContext,
        mcp_workspace: uuid.UUID,
        real_services: Services,
    ) -> None:
        # The user belongs to both workspaces, so only the document lookup's
        # workspace filter refuses the pairing.
        other = await _workspace_of(db_session, mcp_user, "Other")
        foreign = await _indexed_doc(db_session, other, mcp_user, version=1)
        await _chunk(db_session, foreign, version=1, index=0, section=None)

        with pytest.raises(NotFoundError):
            await read_document(
                as_user, workspace_id=mcp_workspace, document_id=foreign.id
            )
