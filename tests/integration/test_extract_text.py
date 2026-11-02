"""Integration tests for extract_text Celery task against real PostgreSQL 18."""

from __future__ import annotations

import io
import sys
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from kombu.exceptions import OperationalError as BrokerError
from pypdf import PdfWriter
from sqlalchemy import create_engine, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.domain.documents import ContentType, DocumentStatus, FailureReason
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.workers.pipeline import PermanentFailure
from tests.pdfs import AWKWARD_TO_UNICODE, pdf_showing, pdf_with_text

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


def _status(session: Session, doc: Document) -> DocumentStatus | None:
    return session.scalar(select(Document.status).where(Document.id == doc.id))


def _spy_reads(
    monkeypatch: pytest.MonkeyPatch, session: Session, doc: Document
) -> list[DocumentStatus | None]:
    """Record the document's status at every attempt's file read."""
    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    real_read = extract_mod._read_file
    seen: list[DocumentStatus | None] = []

    def _spy(file_path: str) -> bytes:
        seen.append(_status(session, doc))
        return real_read(file_path)  # type: ignore[no-any-return]

    monkeypatch.setattr(extract_mod, "_read_file", _spy)
    return seen


def test_a_file_that_stays_unreadable_fails_after_the_retries(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the file touches the outside world, so it is retried; the document
    stays PROCESSING between attempts (D4), and only the last one marks it FAILED."""
    from src.workers.tasks.extract_text import extract_text

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(tmp_path / "nonexistent.txt"))
    reads = _spy_reads(monkeypatch, setup_session, doc)

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert reads == [DocumentStatus.PROCESSING] * 4
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.INTERNAL_ERROR,
    )


def test_a_read_that_fails_once_is_retried_and_succeeds(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "flaky.txt"
    file_path.write_text("there on the second try", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))
    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    real_read = extract_mod._read_file
    calls = 0

    def _fails_once(fp: str) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient I/O error")
        return real_read(fp)  # type: ignore[no-any-return]

    import src.workers.tasks.embed_chunks as embed_mod

    monkeypatch.setattr(extract_mod, "_read_file", _fails_once)
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", MagicMock())

    result = extract_text.apply(args=[str(doc.id)])

    assert (result.state, calls) == ("SUCCESS", 2)
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (DocumentStatus.READY, None)
    assert doc.raw_text == "there on the second try"


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"%PDF-1.4\nnot a pdf at all\n", ContentType.PDF),
        (b"caf\xe9", ContentType.PLAINTEXT),  # latin-1, not UTF-8
    ],
)
def test_a_file_that_cannot_be_parsed_fails_at_once_as_unreadable(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    content_type: ContentType,
) -> None:
    """Parsing is a pure function of the bytes: retrying can only fail again."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "damaged"
    file_path.write_bytes(content)
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), content_type)
    reads = _spy_reads(monkeypatch, setup_session, doc)

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert len(reads) == 1
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.UNREADABLE_FILE,
    )


def test_a_pdf_is_extracted_in_the_child_process(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(pdf_with_text("Quarterly report", "Second page"))
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)
    import src.workers.tasks.embed_chunks as embed_mod

    monkeypatch.setattr(embed_mod.embed_chunks, "delay", MagicMock())

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "SUCCESS"
    setup_session.refresh(doc)
    assert (doc.status, doc.raw_text) == (
        DocumentStatus.READY,
        "Quarterly report\nSecond page",
    )


def test_a_pdf_parse_past_its_timeout_fails_at_once_as_extraction_timeout(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    settings_override: pytest.MonkeyPatch,
) -> None:
    """The bound on the parse is a kill nothing can swallow (S6-10): with a timeout
    shorter than any parse, the document fails, never READY with partial text."""
    from src.core.config import settings
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "slow.pdf"
    file_path.write_bytes(pdf_with_text(*(f"page {i}" for i in range(50))))
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)
    reads = _spy_reads(monkeypatch, setup_session, doc)
    settings_override.setattr(settings, "CELERY_EXTRACT_PARSE_TIMEOUT", 0.01)

    result = extract_text.apply(args=[str(doc.id)])

    assert (result.state, len(reads)) == ("FAILURE", 1)
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason, doc.raw_text) == (
        DocumentStatus.FAILED,
        FailureReason.EXTRACTION_TIMEOUT,
        None,
    )


def _stand_in_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    child = tmp_path / "child.py"
    child.write_text(body)
    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_PDF_CHILD", child)


def test_a_pdf_whose_parse_is_killed_every_time_fails_as_too_large(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SIGKILL we didn't send is the OOM killer under the extract container's cap
    (S8-18): an attempt, retried with the document PROCESSING. Killed on every
    attempt, the file needs more memory than the worker has."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "hungry.pdf"
    file_path.write_bytes(pdf_with_text("needs too much memory"))
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)
    reads = _spy_reads(monkeypatch, setup_session, doc)
    _stand_in_child(
        monkeypatch, tmp_path, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    )

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert reads == [DocumentStatus.PROCESSING] * 4
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason, doc.raw_text) == (
        DocumentStatus.FAILED,
        FailureReason.EXTRACTION_TOO_LARGE,
        None,
    )


def test_a_parse_killed_once_is_retried_and_succeeds(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OOM killer picks the biggest process at that instant, which can be an
    innocent document's parse: its next attempt indexes it."""
    import src.workers.tasks.embed_chunks as embed_mod
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "innocent.pdf"
    file_path.write_bytes(pdf_with_text("an ordinary document"))
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)
    reads = _spy_reads(monkeypatch, setup_session, doc)
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", MagicMock())
    marker = tmp_path / "killed-once"
    _stand_in_child(
        monkeypatch,
        tmp_path,
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        f"marker = Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        "    marker.touch()\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'an ordinary document')\n",
    )

    result = extract_text.apply(args=[str(doc.id)])

    assert (result.state, len(reads)) == ("SUCCESS", 2)
    setup_session.refresh(doc)
    assert (doc.status, doc.raw_text) == (DocumentStatus.READY, "an ordinary document")


def _scanned_pdf(pages: int) -> bytes:
    # What a scanner produces: pages without a text layer.
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        # Extracts to "\n\n\n", not "": an empty-string check would miss it (S6-1).
        (_scanned_pdf(4), ContentType.PDF),
        (" \n\t\u00a0\u2003\n".encode(), ContentType.PLAINTEXT),
        (b"\n\n   \n", ContentType.MARKDOWN),
    ],
    ids=["scanned-pdf", "whitespace-text", "whitespace-markdown"],
)
def test_a_document_without_text_ends_failed_as_no_text(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    content_type: ContentType,
) -> None:
    """D2/D3 end to end: extraction succeeds, there is nothing to index, and the
    document fails as no_text instead of staying READY for good. No embedding call."""
    import src.workers.tasks.embed_chunks as embed_mod
    from src.workers.tasks.embed_chunks import embed_chunks
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "blank"
    file_path.write_bytes(content)
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), content_type)
    dispatched = MagicMock()
    embedding = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", dispatched)
    monkeypatch.setattr(embed_mod, "_run_embedding", embedding)

    extract_text.apply(args=[str(doc.id)])
    dispatched.assert_called_once_with(str(doc.id))
    result = embed_chunks.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    embedding.assert_not_called()
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.NO_TEXT,
    )


@pytest.mark.parametrize(
    ("pdf", "text"),
    [
        (pdf_showing(rb"A\000B"), "A\ufffdB"),
        (
            pdf_showing(b"ABCDE", to_unicode=AWKWARD_TO_UNICODE),
            "\U0001f600\U0001f600\ufffd\ufffd",
        ),
    ],
    ids=["nul-in-a-string", "awkward-to-unicode"],
)
def test_a_pdf_with_unstorable_characters_is_saved_with_replacements(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
    pdf: bytes,
    text: str,
) -> None:
    """S1-3 / S6-2: one unmapped glyph doesn't cost the document; it is READY on the
    first attempt, the unstorable characters replaced by U+FFFD."""
    import src.workers.tasks.embed_chunks as embed_mod
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "glyphs.pdf"
    file_path.write_bytes(pdf)
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)
    reads = _spy_reads(monkeypatch, setup_session, doc)
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", MagicMock())

    result = extract_text.apply(args=[str(doc.id)])

    assert (result.state, len(reads)) == ("SUCCESS", 1)
    setup_session.refresh(doc)
    assert (doc.status, doc.raw_text) == (DocumentStatus.READY, text)


def test_a_failure_that_cannot_be_recorded_is_logged(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the FAILED write itself fails, the document stays non-terminal (the
    reconciler's case, step 9) and the log says so; on_failure doesn't raise."""
    import structlog

    import src.workers.pipeline as pipeline
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "odd.pdf"
    file_path.write_bytes(b"%PDF-1.4\nnot a pdf at all\n")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path), ContentType.PDF)

    def _database_down() -> None:
        raise OperationalError("UPDATE documents", None, Exception("connection lost"))

    monkeypatch.setattr(pipeline, "get_sync_session", _database_down)

    with structlog.testing.capture_logs() as logs:
        result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert {
        "event": "could not record the failure; document left as it was",
        "document_id": str(doc.id),
        "failure_reason": FailureReason.UNREADABLE_FILE,
        "error_type": "OperationalError",
        "log_level": "error",
    }.items() <= next(
        log for log in logs if log["event"].startswith("could not record")
    ).items()
    assert _status(setup_session, doc) == DocumentStatus.PROCESSING


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
    """The soft limit, should it fire during the parse, also means
    extraction_timeout: a backstop, since the child's own timeout comes first."""
    from celery.exceptions import SoftTimeLimitExceeded

    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc_soft.txt"
    file_path.write_text("content", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _timeout(data: bytes, ct: ContentType) -> str:
        raise SoftTimeLimitExceeded("soft limit exceeded")

    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_parse", _timeout)
    reads = _spy_reads(monkeypatch, setup_session, doc)

    extract_text.apply(args=[str(doc.id)])

    assert len(reads) == 1
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.EXTRACTION_TIMEOUT,
    )


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
    """Task re-raises (triggering Celery retry) when _read_file raises OSError."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "doc3.txt"
    file_path.write_text("content", encoding="utf-8")

    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _fail(fp: str) -> bytes:
        raise OSError("simulated read failure")

    import sys

    # Use sys.modules to avoid name collision between the module and the Celery
    # task attribute (both named 'extract_text') that confuses normal import.
    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_read_file", _fail)

    with pytest.raises(OSError):
        extract_text.run(str(doc.id))


# ---------------------------------------------------------------------------
# Guarded transitions: a run never overwrites a status another run moved past
# ---------------------------------------------------------------------------


def _rival_sets(
    session: Session, doc: Document, status: DocumentStatus, raw_text: str | None
) -> None:
    """Commit what a concurrent run (a redelivered or requeued message) would."""
    session.execute(
        update(Document)
        .where(Document.id == doc.id)
        .values(status=status, raw_text=raw_text)
    )
    session.commit()


def test_no_run_starts_on_a_failed_document(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No run starts on FAILED: a late message reads nothing and dispatches nothing."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "failed.txt"
    file_path.write_text("would extract fine now", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))
    _rival_sets(setup_session, doc, DocumentStatus.FAILED, None)

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
    assert doc.raw_text is None
    mock_delay.assert_not_called()


def test_a_failed_extraction_never_overwrites_a_rivals_ready(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction fails only its own stage: a rival's READY survives this failure."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "rival_ready.txt"
    file_path.write_text("content", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _rival_finishes_then_this_run_fails(data: bytes, ct: ContentType) -> str:
        _rival_sets(setup_session, doc, DocumentStatus.READY, "rival text")
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)

    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_parse", _rival_finishes_then_this_run_fails)

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"  # on_failure ran; its guarded write found READY

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert doc.raw_text == "rival text"


def test_a_finished_extraction_never_overwrites_a_rivals_indexed(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY only follows PROCESSING: a rival's INDEXED stays, nothing re-embeds."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "rival_indexed.txt"
    file_path.write_text("content", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    def _rival_indexes_meanwhile(data: bytes, ct: ContentType) -> str:
        _rival_sets(setup_session, doc, DocumentStatus.INDEXED, "rival text")
        return "this run's text"

    import sys

    import src.workers.tasks.embed_chunks as embed_mod

    extract_mod = sys.modules["src.workers.tasks.extract_text"]
    monkeypatch.setattr(extract_mod, "_parse", _rival_indexes_meanwhile)
    mock_delay = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    extract_text.run(str(doc.id))

    setup_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED
    assert doc.raw_text == "rival text"
    mock_delay.assert_not_called()


# ---------------------------------------------------------------------------
# The hand-off: READY means embedding is owed, so a late run dispatches it again
# ---------------------------------------------------------------------------


def test_a_redelivered_extraction_dispatches_embedding_again(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message run again after its READY commit (worker lost before the dispatch)
    sends the hand-off again and extracts nothing: the file is gone, so a read would
    fail the run."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "redelivered.txt"
    file_path.write_text("extracted once", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    assert extract_text.apply(args=[str(doc.id)]).state == "SUCCESS"
    file_path.unlink()
    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "SUCCESS"
    assert mock_delay.call_args_list == [call(str(doc.id))] * 2
    setup_session.refresh(doc)
    assert (doc.status, doc.raw_text) == (DocumentStatus.READY, "extracted once")


def test_a_failed_dispatch_is_retried_without_extracting_again(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker error at the dispatch is retried by autoretry; the retry finds READY
    and dispatches without reading the file again."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "broker_blip.txt"
    file_path.write_text("dispatched on the second try", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))
    reads = _spy_reads(monkeypatch, setup_session, doc)

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock(side_effect=[BrokerError("broker down"), None])
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "SUCCESS"
    assert reads == [DocumentStatus.PROCESSING]
    assert mock_delay.call_count == 2
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason) == (DocumentStatus.READY, None)


def test_a_dispatch_that_keeps_failing_leaves_the_text_ready(
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every attempt tries the dispatch; when none is left, the document stays READY
    with its text (the extraction succeeded; the reconciler re-sends, step 9) and the
    log doesn't blame a rival."""
    import structlog

    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "broker_down.txt"
    file_path.write_text("extracted, never handed off", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))
    reads = _spy_reads(monkeypatch, setup_session, doc)

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock(side_effect=BrokerError("broker down"))
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    with structlog.testing.capture_logs() as logs:
        result = extract_text.apply(args=[str(doc.id)])

    assert result.state == "FAILURE"
    assert reads == [DocumentStatus.PROCESSING]
    assert mock_delay.call_count == 4
    setup_session.refresh(doc)
    assert (doc.status, doc.failure_reason, doc.raw_text) == (
        DocumentStatus.READY,
        None,
        "extracted, never handed off",
    )
    assert any(
        log["event"] == "failure not recorded: the document has moved past this stage"
        for log in logs
    )


@pytest.mark.parametrize("late_on", ["indexed", "deleted"])
def test_a_late_extraction_dispatches_nothing(
    late_on: str,
    tmp_path: Path,
    setup_session: Session,
    patched_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past READY (or gone) nothing is owed: no dispatch. FAILED:
    test_no_run_starts_on_a_failed_document."""
    from src.workers.tasks.extract_text import extract_text

    file_path = tmp_path / "late.txt"
    file_path.write_text("content", encoding="utf-8")
    user = _make_user(setup_session)
    ws = _make_workspace(setup_session, user)
    doc = _make_document(setup_session, ws, user, str(file_path))
    document_id = str(doc.id)
    if late_on == "indexed":
        _rival_sets(setup_session, doc, DocumentStatus.INDEXED, "indexed text")
    else:
        setup_session.delete(doc)
        setup_session.commit()

    import src.workers.tasks.embed_chunks as embed_mod

    mock_delay = MagicMock()
    monkeypatch.setattr(embed_mod.embed_chunks, "delay", mock_delay)

    assert extract_text.apply(args=[document_id]).state == "SUCCESS"
    mock_delay.assert_not_called()
