"""Integration tests for embed_chunks Celery task against real PostgreSQL 18."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.domain.documents import ContentType, DocumentStatus
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace

_DIMS = settings.EMBEDDING_DIMENSIONS


# ---------------------------------------------------------------------------
# Session-scoped sync fixtures (mirrors test_extract_text.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sync_test_db_url(
    postgres_container,  # type: ignore[no-untyped-def]
    apply_migrations: None,
) -> str:
    url: str = postgres_container.get_connection_url()
    return url.replace("psycopg2", "psycopg")


@pytest.fixture(scope="session")
def sync_test_engine(sync_test_db_url: str) -> Generator[Engine]:
    engine = create_engine(sync_test_db_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def sync_test_session_factory(
    sync_test_engine: Engine,
) -> sessionmaker[Session]:
    return sessionmaker(sync_test_engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_db(
    monkeypatch: pytest.MonkeyPatch,
    sync_test_session_factory: sessionmaker[Session],
) -> None:
    import src.workers.database as worker_db

    monkeypatch.setattr(worker_db, "sync_session_factory", sync_test_session_factory)


@pytest.fixture
def setup_session(
    sync_test_session_factory: sessionmaker[Session],
) -> Generator[Session]:
    session = sync_test_session_factory()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Async stand-in for _run_embedding — must be a coroutine because the task
# calls asyncio.run(_run_embedding(...)), which requires an awaitable.
# ---------------------------------------------------------------------------


async def _fake_run_embedding(texts: list[str], **_kw: object) -> list[list[float]]:
    return [[0.0] * _DIMS for _ in texts]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _make_user(session: Session) -> User:
    user = User(
        email=f"embed-test-{uuid.uuid4()}@example.com",
        hashed_password="$2b$12$testhash",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _make_workspace(session: Session, user: User) -> Workspace:
    ws = Workspace(
        name="Embed Test WS",
        slug=f"embed-{uuid.uuid4().hex[:8]}",
        created_by=user.id,
    )
    session.add(ws)
    session.commit()
    session.refresh(ws)
    return ws


def _make_ready_document(
    session: Session,
    workspace: Workspace,
    user: User,
    raw_text: str,
    content_type: ContentType = ContentType.PLAINTEXT,
    version: int = 1,
) -> Document:
    doc = Document(
        workspace_id=workspace.id,
        title="Embed Test Doc",
        content_type=content_type,
        status=DocumentStatus.READY,
        file_path="/dev/null",
        file_size_bytes=len(raw_text),
        uploaded_by=user.id,
        raw_text=raw_text,
        version=version,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_chunks_creates_chunks(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task produces DocumentChunk rows for a READY document."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(
        setup_session, ws, user, raw_text="Hello world. This is a test document."
    )

    with patch(
        "src.workers.tasks.embed_chunks._run_embedding",
        new=_fake_run_embedding,
    ):
        embed_chunks.run(str(doc.id))

    chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(chunks) >= 1
    assert all(c.version == 1 for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_embed_chunks_reindexing_replaces_old_chunks(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Running task again after bumping document version deletes old chunks."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(
        setup_session, ws, user, raw_text="Initial text for version one.", version=1
    )

    with patch(
        "src.workers.tasks.embed_chunks._run_embedding",
        new=_fake_run_embedding,
    ):
        embed_chunks.run(str(doc.id))

    v1_chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(v1_chunks) >= 1

    # Refresh so setup_session sees the INDEXED status the task wrote; then reset
    # to READY to simulate what DocumentService does before re-triggering embed_chunks.
    setup_session.refresh(doc)
    doc.raw_text = "Updated text for version two with more content."
    doc.version = 2
    doc.status = DocumentStatus.READY
    setup_session.commit()

    with patch(
        "src.workers.tasks.embed_chunks._run_embedding",
        new=_fake_run_embedding,
    ):
        embed_chunks.run(str(doc.id))

    all_chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    # All surviving chunks must be version 2 — version 1 chunks deleted.
    assert all(c.version == 2 for c in all_chunks)


def test_embed_chunks_skips_non_ready_document(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task returns without creating chunks when document status is not READY."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = Document(
        workspace_id=ws.id,
        title="Pending Doc",
        content_type=ContentType.PLAINTEXT,
        status=DocumentStatus.PENDING,
        file_path="/dev/null",
        file_size_bytes=10,
        uploaded_by=user.id,
        raw_text=None,
        version=1,
    )
    setup_session.add(doc)
    setup_session.commit()
    setup_session.refresh(doc)

    # No embedding should run — task exits early when status != READY.
    # Asserting zero chunks is sufficient; we don't need a mock call assertion.
    embed_chunks.run(str(doc.id))

    chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(chunks) == 0


def test_hnsw_index_exists(
    sync_test_engine: Engine,
    apply_migrations: None,
) -> None:
    """Verify the HNSW index was created by the migration."""
    with sync_test_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'document_chunks' "
                "AND indexname = 'ix_chunks_embedding_hnsw'"
            )
        )
        rows = result.fetchall()
    assert len(rows) == 1, "HNSW index ix_chunks_embedding_hnsw not found"


def test_embed_chunks_raw_text_none_skips(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task returns early when document is READY but raw_text is None."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = Document(
        workspace_id=ws.id,
        title="No Text",
        content_type=ContentType.PLAINTEXT,
        status=DocumentStatus.READY,
        file_path="/dev/null",
        file_size_bytes=0,
        uploaded_by=user.id,
        raw_text=None,
        version=1,
    )
    setup_session.add(doc)
    setup_session.commit()
    setup_session.refresh(doc)

    embed_chunks.run(str(doc.id))

    chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(chunks) == 0


def test_embed_chunks_document_not_found(
    patched_db: None,
) -> None:
    """Task returns silently when the document ID does not exist."""
    from src.workers.tasks.embed_chunks import embed_chunks

    embed_chunks.run(str(uuid.uuid4()))


def test_embed_chunks_no_chunks_produced(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task returns early and creates no chunks when the chunker produces nothing."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(setup_session, ws, user, raw_text="Some text.")

    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=[])

    with patch("src.workers.tasks.embed_chunks.get_chunker", return_value=mock_chunker):
        embed_chunks.run(str(doc.id))

    chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(chunks) == 0


def test_embed_chunks_version_changed_during_reindex(
    setup_session: Session,
    patched_db: None,
    sync_test_session_factory: sessionmaker[Session],
) -> None:
    """Stale chunks are discarded when doc version is bumped mid-embedding."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(setup_session, ws, user, raw_text="Content to embed.")
    doc_id = doc.id

    async def _bump_and_embed(texts: list[str], **_kw: object) -> list[list[float]]:
        with sync_test_session_factory() as sess:
            d = sess.get(Document, doc_id)
            if d is not None:
                d.version = 999
                sess.commit()
        return [[0.0] * _DIMS for _ in texts]

    with patch("src.workers.tasks.embed_chunks._run_embedding", new=_bump_and_embed):
        embed_chunks.run(str(doc.id))

    chunks = list(
        setup_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars()
    )
    assert len(chunks) == 0


def test_embed_chunks_retry_on_embedding_failure(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Embedding failure re-raises the original exception (called_directly path)."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(setup_session, ws, user, raw_text="Embed this.")

    async def _failing(*args: object, **kwargs: object) -> list[list[float]]:
        raise RuntimeError("API unavailable")

    with (
        patch("src.workers.tasks.embed_chunks._run_embedding", new=_failing),
        pytest.raises(RuntimeError, match="API unavailable"),
    ):
        embed_chunks.run(str(doc.id))


def test_embedding_dimensions_match_settings(
    setup_session: Session,
    patched_db: None,
    sync_test_engine: Engine,
) -> None:
    """Verify stored embedding vectors match EMBEDDING_DIMENSIONS from settings."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_ready_document(
        setup_session, ws, user, raw_text="Dimension check document."
    )

    with patch(
        "src.workers.tasks.embed_chunks._run_embedding",
        new=_fake_run_embedding,
    ):
        embed_chunks.run(str(doc.id))

    with sync_test_engine.connect() as conn:
        # vector_dims() is the pgvector function for checking vector size.
        # array_length() does not work on the vector type.
        row = conn.execute(
            text(
                "SELECT vector_dims(embedding) FROM document_chunks "
                "WHERE document_id = :doc_id LIMIT 1"
            ),
            {"doc_id": doc.id},
        ).fetchone()

    assert row is not None
    assert row[0] == _DIMS
