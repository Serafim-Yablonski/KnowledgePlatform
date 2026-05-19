"""Integration tests for SQLAlchemySearchRepository against real PostgreSQL 18."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import ContentType, DocumentStatus
from src.models.chunk import EMBEDDING_DIMS, DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.search import SQLAlchemySearchRepository
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.auth import UserCreate

# ---------------------------------------------------------------------------
# Deterministic unit-vector embeddings for testing cosine similarity ranking.
# v1 points in dim 0, v2 points in dim 1.
# A query [0.9, 0.1, 0, ...] has cosine similarity ~0.993 to v1 and ~0.110 to v2.
# ---------------------------------------------------------------------------
_V1: list[float] = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)
_V2: list[float] = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMS - 2)
_QUERY_NEAR_V1: list[float] = [0.9, 0.1] + [0.0] * (EMBEDDING_DIMS - 2)
_QUERY_NEAR_V2: list[float] = [0.1, 0.9] + [0.0] * (EMBEDDING_DIMS - 2)
_LOW_SIM: list[float] = [0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIMS - 3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    return await repo.create(
        UserCreate(
            email=f"search-int-{uuid.uuid4()}@example.com", password="password123"
        ),
        hashed_password="$2b$12$testhash",
    )


@pytest.fixture
async def test_workspace(db_session: AsyncSession, test_user: User) -> Workspace:
    repo = SQLAlchemyWorkspaceRepository(db_session)
    ws = await repo.create(
        name=f"Search WS {uuid.uuid4().hex[:6]}",
        slug=f"search-ws-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await db_session.flush()
    return ws


async def _create_ready_doc(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    uploaded_by: uuid.UUID,
    title: str = "Test Doc",
    version: int = 1,
) -> Document:
    doc = Document(
        workspace_id=workspace_id,
        title=title,
        content_type=ContentType.PLAINTEXT,
        file_path=f"/tmp/{uuid.uuid4()}.txt",
        file_size_bytes=100,
        uploaded_by=uploaded_by,
        status=DocumentStatus.READY,
        version=version,
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


async def _create_chunk(
    session: AsyncSession,
    document_id: uuid.UUID,
    embedding: list[float],
    text: str = "chunk text",
    version: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding,
        token_count=len(text.split()),
        metadata_={},
        version=version,
    )
    session.add(chunk)
    await session.flush()
    await session.refresh(chunk)
    return chunk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_similar_chunk_ranks_first(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Chunk with embedding closest to the query should rank first."""
    doc = await _create_ready_doc(db_session, test_workspace.id, test_user.id)
    chunk_v1 = await _create_chunk(
        db_session, doc.id, _V1, text="dimension zero content"
    )
    chunk_v2 = await _create_chunk(
        db_session, doc.id, _V2, text="dimension one content", chunk_index=1
    )

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=5,
        min_score=0.0,
    )

    assert len(results) == 2
    assert results[0].chunk_id == chunk_v1.id
    assert results[1].chunk_id == chunk_v2.id
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_workspace_isolation(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Chunks from workspace B must not appear in workspace A's results."""
    ws_b = await SQLAlchemyWorkspaceRepository(db_session).create(
        name=f"WS-B {uuid.uuid4().hex[:4]}",
        slug=f"ws-b-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await db_session.flush()

    doc_a = await _create_ready_doc(db_session, test_workspace.id, test_user.id)
    doc_b = await _create_ready_doc(db_session, ws_b.id, test_user.id)

    chunk_a = await _create_chunk(db_session, doc_a.id, _V1, text="workspace A chunk")
    await _create_chunk(db_session, doc_b.id, _V1, text="workspace B chunk")

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=10,
        min_score=0.0,
    )

    result_ids = {r.chunk_id for r in results}
    assert chunk_a.id in result_ids
    assert len(results) == 1  # only workspace A's chunk


@pytest.mark.asyncio
async def test_version_filtering_excludes_stale_chunks(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Chunks with version < document.version must not be returned."""
    doc = await _create_ready_doc(
        db_session, test_workspace.id, test_user.id, version=2
    )

    stale_chunk = await _create_chunk(
        db_session, doc.id, _V1, text="stale chunk", version=1
    )
    current_chunk = await _create_chunk(
        db_session, doc.id, _V1, text="current chunk", version=2, chunk_index=1
    )

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=10,
        min_score=0.0,
    )

    result_ids = {r.chunk_id for r in results}
    assert current_chunk.id in result_ids
    assert stale_chunk.id not in result_ids


@pytest.mark.asyncio
async def test_min_score_filters_low_similarity(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Chunks with similarity below min_score must be excluded."""
    doc = await _create_ready_doc(db_session, test_workspace.id, test_user.id)

    # _V1 is very similar to _QUERY_NEAR_V1 (score ~0.99)
    high_chunk = await _create_chunk(db_session, doc.id, _V1, text="high sim")
    # _LOW_SIM is nearly orthogonal to _QUERY_NEAR_V1 (score ~0.0)
    await _create_chunk(db_session, doc.id, _LOW_SIM, text="low sim", chunk_index=1)

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=10,
        min_score=0.5,
    )

    result_ids = {r.chunk_id for r in results}
    assert high_chunk.id in result_ids
    assert len(results) == 1


@pytest.mark.asyncio
async def test_non_ready_documents_excluded(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Chunks belonging to PENDING/PROCESSING/FAILED documents must not appear."""
    pending_doc = Document(
        workspace_id=test_workspace.id,
        title="Pending",
        content_type=ContentType.PLAINTEXT,
        file_path="/tmp/pending.txt",
        file_size_bytes=10,
        uploaded_by=test_user.id,
        status=DocumentStatus.PENDING,
        version=1,
    )
    db_session.add(pending_doc)
    await db_session.flush()

    await _create_chunk(db_session, pending_doc.id, _V1, text="should not appear")

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=10,
        min_score=0.0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_top_k_limits_results(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    doc = await _create_ready_doc(db_session, test_workspace.id, test_user.id)
    for i in range(5):
        await _create_chunk(db_session, doc.id, _V1, chunk_index=i)

    repo = SQLAlchemySearchRepository(db_session)
    results = await repo.search_similar(
        workspace_id=test_workspace.id,
        embedding=_QUERY_NEAR_V1,
        top_k=3,
        min_score=0.0,
    )

    assert len(results) == 3
