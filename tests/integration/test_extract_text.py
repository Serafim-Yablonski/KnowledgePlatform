"""Integration tests for extract_text Celery task against real PostgreSQL 18."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.domain.documents import ContentType, DocumentStatus
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sync_test_db_url(
    postgres_container,  # type: ignore[no-untyped-def]
    apply_migrations: None,
) -> str:
    url: str = postgres_container.get_connection_url()
    # testcontainers returns a psycopg2 URL; convert to psycopg3 for the sync engine.
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
    """Session for test setup — commits directly to the testcontainer DB."""
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
        email=f"celery-test-{uuid.uuid4()}@example.com",
        hashed_password="$2b$12$testhash",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _make_workspace(session: Session, user: User) -> Workspace:
    ws = Workspace(
        name="Celery Test WS",
        slug=f"celery-{uuid.uuid4().hex[:8]}",
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
    file_path: str,
    content_type: ContentType = ContentType.PLAINTEXT,
) -> Document:
    doc = Document(
        workspace_id=workspace.id,
        title="Test Doc",
        content_type=content_type,
        status=DocumentStatus.PENDING,
        file_path=file_path,
        file_size_bytes=len(file_path),
        uploaded_by=user.id,
        version=1,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_plaintext(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task populates raw_text from a UTF-8 text file and sets status READY."""
    from src.workers.tasks.extract_text import extract_text

    content = "Hello from the integration test"
    file_path = tmp_path / "doc.txt"
    file_path.write_text(content, encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert doc.raw_text == content


def test_extract_pdf(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task sets status READY for a valid (blank) PDF."""
    from src.workers.tasks.extract_text import extract_text

    pdf_path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as fh:
        writer.write(fh)

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(pdf_path), ContentType.PDF)

    extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert doc.raw_text is not None


def test_extract_idempotent(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
) -> None:
    """Running the task twice on a READY document is a no-op."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc2.txt"
    file_path.write_text("Idempotency test", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    extract_text.run(str(doc.id))
    extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY


def test_extract_retries_processing_status(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
) -> None:
    """PROCESSING documents are re-processed (crash-recovery for a killed worker).

    The task intentionally does not skip PROCESSING — a worker that set the
    status and then crashed must be retryable without manual intervention.
    """
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc_proc.txt"
    file_path.write_text("Processing guard test", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    # Simulate a prior worker crash: status is stuck at PROCESSING.
    doc.status = DocumentStatus.PROCESSING
    setup_session.commit()

    with patch("src.workers.tasks.embed_chunks.embed_chunks"):
        extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    # Task must complete and advance the document to READY.
    assert doc.status == DocumentStatus.READY


def test_extract_missing_file_sets_failed(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task sets status FAILED and re-raises when the file does not exist."""
    from src.workers.tasks.extract_text import extract_text

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(tmp_path / "nonexistent.txt"))

    with pytest.raises(OSError):
        extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_extract_document_not_found_returns_silently(
    setup_session: Session,
    patched_db: None,
) -> None:
    """Task logs a warning and returns without error when the document is missing."""
    from src.workers.tasks.extract_text import extract_text

    # Random UUID that has no corresponding DB row.
    extract_text.run(str(uuid.uuid4()))


def test_extract_soft_time_limit_marks_failed(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SoftTimeLimitExceeded during PDF parsing marks document FAILED instead of
    leaving it stuck in PROCESSING."""
    from celery.exceptions import SoftTimeLimitExceeded

    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc_soft.txt"
    file_path.write_text("content", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _timeout(fp: str, ct: ContentType) -> str:
        raise SoftTimeLimitExceeded("soft limit exceeded")

    import sys

    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_read_text", _timeout)

    with pytest.raises(SoftTimeLimitExceeded):
        extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_extract_dispatches_embed(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful extraction dispatches embed_chunks with the document id."""
    from unittest.mock import MagicMock

    from src.workers.tasks.extract_text import extract_text

    content = "dispatch test content"
    file_path = tmp_path / "doc_dispatch.txt"
    file_path.write_text(content, encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    extract_text.run(str(doc.id))

    mock_delay.assert_called_once_with(str(doc.id))


def test_extract_retry_on_io_error(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task re-raises (triggering Celery retry) when _read_text raises OSError."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc3.txt"
    file_path.write_text("content", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _fail(fp: str, ct: ContentType) -> str:
        raise OSError("simulated read failure")

    import sys

    # Use sys.modules to avoid name collision between the module and the Celery
    # task attribute (both named 'extract_text') that confuses normal import.
    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_read_text", _fail)

    with pytest.raises(OSError):
        extract_text.run(str(doc.id))
