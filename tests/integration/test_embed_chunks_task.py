"""Integration tests for embed_chunks Celery task against real PostgreSQL 18."""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.ai.chunking.factory import get_chunker
from src.core.config import settings
from src.domain.documents import ContentType, DocumentStatus, FailureReason
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.workers.pipeline import EmbeddingError

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


def _chunk_indexes(session: Session, doc: Document) -> list[int]:
    return sorted(
        session.scalars(
            sa.select(DocumentChunk.chunk_index).where(
                DocumentChunk.document_id == doc.id
            )
        )
    )


def test_racing_runs_leave_one_chunk_set(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs on one document, both past the READY check, leave one chunk set."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    assert doc.raw_text is not None
    expected = len(
        get_chunker(ContentType.PLAINTEXT).chunk(
            doc.raw_text, {"document_id": str(doc.id)}
        )
    )
    rival_ran = False

    async def _embed_after_a_rival(texts: list[str], **_: Any) -> list[list[float]]:
        # The first run, mid-embedding, lets a rival (a redelivered or requeued
        # message) run start to finish; then it swaps too.
        nonlocal rival_ran
        if not rival_ran:
            rival_ran = True
            with ThreadPoolExecutor(1) as pool:
                pool.submit(embed_chunks.run, str(doc.id)).result()
        return _fake_embeddings(texts, settings.EMBEDDING_DIMENSIONS)

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _embed_after_a_rival)

    embed_chunks.run(str(doc.id))

    assert rival_ran
    assert _chunk_indexes(setup_session, doc) == list(range(expected))


def _new_chunks(doc: Document, tag: str) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            text=f"{tag} {i}",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
            token_count=2,
        )
        for i in range(3)
    ]


def _wait_for_a_lock_wait(engine: Engine) -> None:
    # A fresh connection per poll: a transaction keeps its pg_stat_activity snapshot.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            waiting = conn.scalar(
                sa.text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                )
            )
        if waiting:
            return
        time.sleep(0.05)
    pytest.fail("the second swap never waited on a lock")


def test_simultaneous_swaps_take_turns(
    setup_session: Session,
    sync_test_engine: Engine,
    sync_test_session_factory: sessionmaker[Session],
) -> None:
    """A swap that starts while another is uncommitted waits, then replaces its set."""
    from src.workers.tasks.embed_chunks import _replace_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    def _second_swap() -> bool:
        with sync_test_session_factory() as session:
            replaced = _replace_chunks(session, doc.id, _new_chunks(doc, "second"))
            session.commit()
            return replaced

    with (
        sync_test_session_factory() as first,
        ThreadPoolExecutor(1) as pool,
    ):
        assert _replace_chunks(first, doc.id, _new_chunks(doc, "first"))
        second = pool.submit(_second_swap)
        _wait_for_a_lock_wait(sync_test_engine)
        first.commit()
        assert second.result(timeout=10)

    texts = setup_session.scalars(
        sa.select(DocumentChunk.text)
        .where(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index)
    ).all()
    assert texts == ["second 0", "second 1", "second 2"]


def test_a_second_chunk_at_one_position_is_rejected(
    setup_session: Session,
) -> None:
    """The database itself refuses a second chunk set (schema from the migrations)."""
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    setup_session.add_all(_new_chunks(doc, "one"))
    setup_session.commit()

    setup_session.add(_new_chunks(doc, "two")[0])
    with pytest.raises(IntegrityError, match="uq_chunks_document_chunk_index"):
        setup_session.commit()
    setup_session.rollback()


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
    """READY always has raw_text (extraction writes both in one UPDATE): one without
    it is a bug, so it fails as internal_error instead of staying READY for good."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, raw_text=None)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.INTERNAL_ERROR,
    )


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

    with pytest.raises(EmbeddingError):
        embed_chunks.run(str(doc.id))

    # Status must not advance — document stays READY for the retry attempt.
    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY


def test_embed_chunks_sets_failed_when_retries_exhausted(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding calls the outside world, so it is retried; the document stays READY
    between attempts, and only the last one marks it FAILED as embedding_failed."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    seen: list[DocumentStatus | None] = []

    async def _mock_embed_fails(*_: Any, **__: Any) -> list[list[float]]:
        seen.append(_status(setup_session, doc))
        raise RuntimeError("persistent API failure")

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _mock_embed_fails)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert seen == [DocumentStatus.READY] * 4
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.EMBEDDING_FAILED,
    )


def _status(session: Session, doc: Document) -> DocumentStatus | None:
    return session.scalar(sa.select(Document.status).where(Document.id == doc.id))


def test_an_embedding_failure_once_is_retried_and_indexes(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    calls = 0

    async def _fails_once(texts: list[str], **_: Any) -> list[list[float]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("503 from the provider")
        return _fake_embeddings(texts, settings.EMBEDDING_DIMENSIONS)

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _fails_once)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert (result.state, calls) == ("SUCCESS", 2)
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (DocumentStatus.INDEXED, None)


@pytest.mark.parametrize(
    "chunker_error",
    [ValueError("chunker bug"), SoftTimeLimitExceeded()],
    ids=["exception", "soft-limit"],
)
def test_a_chunking_failure_fails_at_once_without_embedding(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    chunker_error: Exception,
) -> None:
    """Chunking is a pure function of the stored text: retrying can only fail again
    (S6-7: it used to sit outside the try and leave the document READY)."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)
    broken = MagicMock()
    broken.chunk.side_effect = chunker_error
    embedded = MagicMock()

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "get_chunker", lambda _ct: broken)
    monkeypatch.setattr(mod, "_run_embedding", embedded)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert broken.chunk.call_count == 1
    embedded.assert_not_called()
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.INTERNAL_ERROR,
    )


def test_a_swap_that_hits_a_database_error_is_retried(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The swap touches the database, so it is retried (re-embedding: the vectors
    aren't kept between attempts)."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _embed(texts: list[str], **_: Any) -> list[list[float]]:
        return _fake_embeddings(texts, settings.EMBEDDING_DIMENSIONS)

    import src.workers.tasks.embed_chunks as mod

    real_replace = mod._replace_chunks
    swaps = 0

    def _fails_once(session: Session, *args: Any) -> bool:
        nonlocal swaps
        swaps += 1
        if swaps == 1:
            raise OperationalError("SELECT ... FOR UPDATE", None, Exception("reset"))
        return real_replace(session, *args)  # type: ignore[no-any-return]

    monkeypatch.setattr(mod, "_run_embedding", _embed)
    monkeypatch.setattr(mod, "_replace_chunks", _fails_once)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert (result.state, swaps) == ("SUCCESS", 2)
    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED


def test_a_losing_runs_failure_never_overwrites_a_winners_indexed(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F5-13: a run that runs out of retries after a rival indexed leaves INDEXED."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _rival_indexes_then_this_run_fails(
        *_: Any, **__: Any
    ) -> list[list[float]]:
        setup_session.execute(
            sa.update(Document)
            .where(Document.id == doc.id)
            .values(status=DocumentStatus.INDEXED)
        )
        setup_session.commit()
        raise RuntimeError("persistent API failure")

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _rival_indexes_then_this_run_fails)
    # The first failure is the last attempt: on_failure runs right after the rival.
    monkeypatch.setattr(mod.embed_chunks, "max_retries", 0)

    result = embed_chunks.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"  # on_failure ran; its guarded write found INDEXED

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED


def test_a_failure_write_waiting_on_a_swap_rechecks_and_changes_nothing(
    setup_session: Session,
    sync_test_engine: Engine,
    sync_test_session_factory: sessionmaker[Session],
) -> None:
    """The guard holds under real concurrency, not only in sequence: the failure
    write waits on the swap's row lock, then re-checks the status of the committed row.
    """
    from src.domain.documents import FAIL_EMBEDDING
    from src.workers.tasks.embed_chunks import _replace_chunks
    from src.workers.transitions import guarded_update

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    def _losers_failure_write() -> bool:
        with sync_test_session_factory() as session:
            moved = session.execute(
                guarded_update(doc.id, FAIL_EMBEDDING).returning(Document.id)
            ).first()
            session.commit()
            return moved is not None

    with (
        sync_test_session_factory() as winner,
        ThreadPoolExecutor(1) as pool,
    ):
        assert _replace_chunks(winner, doc.id, _new_chunks(doc, "winner"))
        loser = pool.submit(_losers_failure_write)
        _wait_for_a_lock_wait(sync_test_engine)
        winner.commit()
        assert loser.result(timeout=10) is False

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED
    assert _chunk_indexes(setup_session, doc) == [0, 1, 2]


def test_a_run_under_way_still_finishes_indexed_over_a_rivals_failed(
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A success beats a failure: the swap sets INDEXED from any status, and the
    failure's reason goes with it."""
    from src.workers.tasks.embed_chunks import embed_chunks

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user)

    async def _rival_fails_meanwhile(texts: list[str], **_: Any) -> list[list[float]]:
        setup_session.execute(
            sa.update(Document)
            .where(Document.id == doc.id)
            .values(
                status=DocumentStatus.FAILED,
                failure_reason=FailureReason.EMBEDDING_FAILED,
            )
        )
        setup_session.commit()
        return _fake_embeddings(texts, settings.EMBEDDING_DIMENSIONS)

    import src.workers.tasks.embed_chunks as mod

    monkeypatch.setattr(mod, "_run_embedding", _rival_fails_meanwhile)

    embed_chunks.run(str(doc.id))

    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (DocumentStatus.INDEXED, None)
    assert _chunk_indexes(setup_session, doc)
