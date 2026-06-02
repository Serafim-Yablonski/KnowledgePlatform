"""Integration tests for embed_chunks Celery task against real PostgreSQL 18."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.domain.documents import ContentType, DocumentStatus
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace

# ---------------------------------------------------------------------------
# Session-scoped fixtures (mirrors test_extract_text.py conventions)
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
    """Replace the Celery worker's session factory with the test one."""
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


def _make_document(
    session: Session,
    workspace: Workspace,
    user: User,
    *,
    status: DocumentStatus = DocumentStatus.READY,
    raw_text: str | None = "Hello world chunk one. Hello world chunk two.",
    version: int = 1,
) -> Document:
    doc = Document(
        workspace_id=workspace.id,
        title="Embed Test Doc",
        content_type=ContentType.PLAINTEXT,
        status=status,
        file_path=f"/data/uploads/{uuid.uuid4()}.txt",
        file_size_bytes=100,
        uploaded_by=user.id,
        raw_text=raw_text,
        version=version,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def _fake_embeddings(texts: list[str], dimensions: int) -> list[list[float]]:
    return [[0.1] * dimensions] * len(texts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_chunks_normal_path(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks are stored and document status transitions to INDEXED."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _mock_embed(texts: list[str], **_: Any) -> list[list[float]]:
        return _fake_embeddings(texts, 768)

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _mock_embed)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED

    chunks = (
        setup_session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        )
        .scalars()
        .all()
    )
    assert len(chunks) > 0
    assert all(c.version == doc.version for c in chunks)


def test_embed_chunks_document_not_found(
    patched_db: None,
) -> None:
    """Task returns silently when the document UUID does not exist."""
    from src.workers.tasks.embed_chunks import embed_chunks

    embed_chunks.run(str(uuid.uuid4()))


def test_embed_chunks_document_not_ready(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task skips a document that is not in READY status."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, status=DocumentStatus.PENDING)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.PENDING


def test_embed_chunks_already_indexed_is_idempotent(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task is a no-op when the document is already INDEXED."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, status=DocumentStatus.INDEXED)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED


def test_embed_chunks_no_raw_text(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task skips a READY document that has no extracted text."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, raw_text=None)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY


def test_embed_chunks_version_conflict_discards_stale_chunks(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    sync_test_session_factory: sessionmaker[Session],
) -> None:
    """Chunks are discarded when the document version changes during embedding.

    Simulates a concurrent re-index: _run_embedding bumps the document version
    while embeddings are being computed so that the second session's version
    check detects the mismatch and rolls back.
    """
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    doc_id = doc.id

    async def _mock_embed_and_bump_version(
        texts: list[str], **_: Any
    ) -> list[list[float]]:
        # Simulate a concurrent re-index bumping the version while we embed.
        with sync_test_session_factory() as bump_session:
            bump_session.execute(
                sa.update(Document)
                .where(Document.id == doc_id)
                .values(version=Document.version + 1)
            )
            bump_session.commit()
        return _fake_embeddings(texts, 768)

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _mock_embed_and_bump_version)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    # Version was bumped by the concurrent update; status must not be INDEXED.
    assert doc.status == DocumentStatus.READY

    chunks = (
        setup_session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        )
        .scalars()
        .all()
    )
    assert len(chunks) == 0


def test_embed_chunks_retry_on_embedding_failure(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task re-raises the original exception (triggering Celery retry) on failure."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _mock_embed_fails(*_: Any, **__: Any) -> list[list[float]]:
        raise RuntimeError("embedding API unavailable")

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _mock_embed_fails)

    with pytest.raises(RuntimeError, match="embedding API unavailable"):
        embed_chunks.run(str(doc.id))

    # Status must not advance — document stays READY for the retry attempt.
    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY


def test_embed_chunks_sets_failed_when_retries_exhausted(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document transitions to FAILED when all retries are exhausted."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _mock_embed_fails(*_: Any, **__: Any) -> list[list[float]]:
        raise RuntimeError("persistent API failure")

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _mock_embed_fails)
    # max_retries=0 means the first failure is already the last attempt.
    monkeypatch.setattr(mod.embed_chunks, "max_retries", 0)

    with pytest.raises(RuntimeError, match="persistent API failure"):
        embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
