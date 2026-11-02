"""The pipeline's jobs (src/workers/jobs.py) on a real PostgreSQL: what each does
with a claimed document. What the worker does with how a job ends (retry, budget,
FAILED) is the runner's, tested with it."""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import sqlalchemy as sa
from pypdf import PdfWriter
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.ai.chunking.factory import get_chunker
from src.core.config import settings
from src.domain.documents import ContentType, DocumentStatus, FailureReason
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.workers import jobs
from src.workers.jobs import embed_chunks, extract_text, replace_chunks
from src.workers.pipeline import EmbeddingError, ParseKilled, PermanentFailure
from src.workers.queue import EMBEDDING, fail
from tests.pdfs import AWKWARD_TO_UNICODE, pdf_showing, pdf_with_text

PARSE_TIMEOUT = 30.0
TEXT = "Hello world chunk one. Hello world chunk two."

Sessions = async_sessionmaker[AsyncSession]


@pytest.fixture
def sessions(test_engine: AsyncEngine) -> Sessions:
    return async_sessionmaker(test_engine, expire_on_commit=False)


async def _document(
    sessions: Sessions,
    *,
    status: DocumentStatus = DocumentStatus.PROCESSING,
    content_type: ContentType = ContentType.PLAINTEXT,
    file_path: str = "/nonexistent",
    raw_text: str | None = None,
) -> uuid.UUID:
    """A committed document, claimed (attempt 1, leased) unless ``status`` says
    otherwise."""
    async with sessions.begin() as s:
        user = User(email=f"jobs-{uuid.uuid4()}@example.com", hashed_password="x")
        s.add(user)
        await s.flush()
        ws = Workspace(
            name="W", slug=f"jobs-{uuid.uuid4().hex[:8]}", created_by=user.id
        )
        s.add(ws)
        await s.flush()
        doc = Document(
            workspace_id=ws.id,
            title="T",
            content_type=content_type,
            status=status,
            file_path=file_path,
            file_size_bytes=1,
            uploaded_by=user.id,
            raw_text=raw_text,
            attempts=1,
            available_at=sa.func.now() + sa.text("interval '5 minutes'"),
        )
        s.add(doc)
        await s.flush()
        return doc.id


async def _row(sessions: Sessions, doc: uuid.UUID) -> sa.Row:  # type: ignore[type-arg]
    async with sessions() as s:
        return (
            await s.execute(
                sa.select(
                    Document.status,
                    Document.failure_reason,
                    Document.raw_text,
                    Document.attempts,
                    (Document.available_at <= sa.func.now()).label("claimable"),
                ).where(Document.id == doc)
            )
        ).one()


async def _chunk_texts(sessions: Sessions, doc: uuid.UUID) -> list[str]:
    async with sessions() as s:
        return list(
            (
                await s.execute(
                    sa.select(DocumentChunk.text)
                    .where(DocumentChunk.document_id == doc)
                    .order_by(DocumentChunk.chunk_index)
                )
            ).scalars()
        )


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


async def _extract(
    sessions: Sessions, doc: uuid.UUID, timeout: float = PARSE_TIMEOUT
) -> None:
    await extract_text(sessions, doc, parse_timeout=timeout)


def _stand_in_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    child = tmp_path / "child.py"
    child.write_text(body)
    monkeypatch.setattr(jobs, "_PDF_CHILD", child)


def _pid_file_child(tmp_path: Path, then: str) -> tuple[Path, str]:
    """A child that records its pid, then runs ``then``."""
    pid_file = tmp_path / "child.pid"
    return (
        pid_file,
        f"import os, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        f"{then}\n",
    )


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_text_is_saved_ready_and_handed_to_embedding_at_once(
    sessions: Sessions, tmp_path: Path
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain notes")
    doc = await _document(sessions, file_path=str(path))

    await _extract(sessions, doc)

    # Embedding may claim it now, with a fresh budget.
    assert tuple(await _row(sessions, doc)) == (
        DocumentStatus.READY,
        None,
        "plain notes",
        0,
        True,
    )


async def test_a_pdf_is_parsed_in_the_child_process(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(pdf_with_text("Quarterly report", "Second page"))
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)

    await _extract(sessions, doc)
    assert (await _row(sessions, doc)).raw_text == "Quarterly report\nSecond page"

    # Whatever the child prints is the text: the parse happens there, not here.
    _stand_in_child(monkeypatch, tmp_path, "print('from the child', end='')")
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)
    await _extract(sessions, doc)
    assert (await _row(sessions, doc)).raw_text == "from the child"


@pytest.mark.parametrize(
    ("content", "content_type"),
    [(b"\xff\xfe not utf-8", ContentType.PLAINTEXT), (b"not a pdf", ContentType.PDF)],
    ids=["bad-utf8", "bad-pdf"],
)
async def test_a_file_that_cannot_be_parsed_fails_permanently_as_unreadable(
    sessions: Sessions, tmp_path: Path, content: bytes, content_type: ContentType
) -> None:
    path = tmp_path / "broken"
    path.write_bytes(content)
    doc = await _document(sessions, file_path=str(path), content_type=content_type)

    with pytest.raises(PermanentFailure) as exc_info:
        await _extract(sessions, doc)

    assert exc_info.value.reason == FailureReason.UNREADABLE_FILE
    assert (await _row(sessions, doc))[:3] == (DocumentStatus.PROCESSING, None, None)


async def test_a_parse_past_its_timeout_fails_permanently_and_its_child_is_gone(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound on the parse is a kill nothing can swallow (S6-10)."""
    pid_file, body = _pid_file_child(tmp_path, "time.sleep(60)")
    _stand_in_child(monkeypatch, tmp_path, body)
    path = tmp_path / "slow.pdf"
    path.write_bytes(b"%PDF")
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)

    with pytest.raises(PermanentFailure) as exc_info:
        await _extract(sessions, doc, timeout=1.0)

    assert exc_info.value.reason == FailureReason.EXTRACTION_TIMEOUT
    assert _gone(int(pid_file.read_text()))


async def test_a_cancelled_extraction_takes_its_child_with_it(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop or the attempt's timeout cancels the job; the parse must not live on."""
    pid_file, body = _pid_file_child(tmp_path, "time.sleep(60)")
    _stand_in_child(monkeypatch, tmp_path, body)
    path = tmp_path / "slow.pdf"
    path.write_bytes(b"%PDF")
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)

    task = asyncio.create_task(_extract(sessions, doc))
    async with asyncio.timeout(10):
        while not pid_file.exists() or not pid_file.read_text():
            await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _gone(int(pid_file.read_text()))


@pytest.mark.parametrize(
    ("body", "raised", "reason"),
    [
        # A SIGKILL we didn't send: the OOM killer (S8-18). An attempt, not a verdict.
        ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", ParseKilled, None),
        # A crash the same bytes repeat: the file's.
        ("import os; os.abort()", PermanentFailure, FailureReason.UNREADABLE_FILE),
        (
            "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
            PermanentFailure,
            FailureReason.UNREADABLE_FILE,
        ),
        # pypdf raised.
        ("import sys; sys.exit(3)", PermanentFailure, FailureReason.UNREADABLE_FILE),
        # Ours (a failed import, say): transient.
        ("import sys; sys.exit(1)", RuntimeError, None),
    ],
    ids=["oom-kill", "abort", "segfault", "parse-failed", "our-bug"],
)
async def test_how_the_child_ends_says_whose_fault_it_is(
    sessions: Sessions,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    raised: type[Exception],
    reason: FailureReason | None,
) -> None:
    _stand_in_child(monkeypatch, tmp_path, body)
    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF")
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)

    with pytest.raises(raised) as exc_info:
        await _extract(sessions, doc)

    assert type(exc_info.value) is raised
    if reason is not None:
        assert exc_info.value.reason == reason  # type: ignore[attr-defined]
    assert (await _row(sessions, doc)).status == DocumentStatus.PROCESSING


@pytest.mark.parametrize(
    ("pdf", "text"),
    [
        (pdf_showing(rb"A\000B"), "A�B"),
        (
            pdf_showing(b"ABCDE", to_unicode=AWKWARD_TO_UNICODE),
            "\U0001f600\U0001f600��",
        ),
    ],
    ids=["nul-in-a-string", "awkward-to-unicode"],
)
async def test_a_pdf_with_unstorable_characters_is_saved_with_replacements(
    sessions: Sessions, tmp_path: Path, pdf: bytes, text: str
) -> None:
    """S1-3 / S6-2: one unmapped glyph doesn't cost the document."""
    path = tmp_path / "glyphs.pdf"
    path.write_bytes(pdf)
    doc = await _document(sessions, file_path=str(path), content_type=ContentType.PDF)

    await _extract(sessions, doc)

    assert (await _row(sessions, doc))[:3] == (DocumentStatus.READY, None, text)


async def test_a_file_that_cannot_be_read_is_transient(sessions: Sessions) -> None:
    doc = await _document(sessions, file_path="/nonexistent/upload.txt")

    with pytest.raises(OSError):
        await _extract(sessions, doc)

    assert (await _row(sessions, doc)).status == DocumentStatus.PROCESSING


async def test_extracting_a_document_that_is_gone_does_nothing(
    sessions: Sessions,
) -> None:
    await _extract(sessions, uuid.uuid4())


@pytest.mark.parametrize(
    "moved_to", [DocumentStatus.READY, DocumentStatus.INDEXED, DocumentStatus.FAILED]
)
async def test_a_finished_extraction_never_overwrites_a_document_that_moved_on(
    sessions: Sessions, tmp_path: Path, moved_to: DocumentStatus
) -> None:
    """A run that outlived its lease finishes after a rival's: the rival's stands."""
    path = tmp_path / "notes.txt"
    path.write_text("late text")
    doc = await _document(
        sessions, file_path=str(path), status=moved_to, raw_text="rival"
    )
    before = tuple(await _row(sessions, doc))

    await _extract(sessions, doc)

    assert tuple(await _row(sessions, doc)) == before


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Answers with constant vectors; ``during`` runs inside the call, as if
    something happened while the provider worked."""

    def __init__(
        self,
        during: Callable[[], Awaitable[None]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls = 0
        self._during = during
        self._error = error

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._during is not None:
            during, self._during = self._during, None
            await during()
        if self._error is not None:
            raise self._error
        return [[0.1] * settings.EMBEDDING_DIMENSIONS for _ in texts]


def _expected_chunks(doc: uuid.UUID, text: str = TEXT) -> list[str]:
    return [
        c.text
        for c in get_chunker(ContentType.PLAINTEXT).chunk(
            text, {"document_id": str(doc)}
        )
    ]


async def test_embedding_indexes_the_document_with_its_chunk_set(
    sessions: Sessions,
) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)
    embedder = FakeEmbedder()

    await embed_chunks(sessions, embedder, doc)

    assert (await _row(sessions, doc))[:2] == (DocumentStatus.INDEXED, None)
    assert await _chunk_texts(sessions, doc) == _expected_chunks(doc)
    async with sessions() as s:
        vector = (
            await s.execute(
                sa.select(DocumentChunk.embedding)
                .where(DocumentChunk.document_id == doc)
                .limit(1)
            )
        ).scalar_one()
    assert len(vector) == settings.EMBEDDING_DIMENSIONS


async def test_embedding_again_replaces_the_chunk_set(sessions: Sessions) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text="old text")
    await embed_chunks(sessions, FakeEmbedder(), doc)
    async with sessions.begin() as s:
        await s.execute(
            sa.update(Document)
            .where(Document.id == doc)
            .values(status=DocumentStatus.READY, raw_text=TEXT)
        )

    await embed_chunks(sessions, FakeEmbedder(), doc)

    assert await _chunk_texts(sessions, doc) == _expected_chunks(doc)


async def test_racing_runs_leave_one_chunk_set(sessions: Sessions) -> None:
    """A rival run (its lease ran out on the first) finishes while the first is
    still embedding; then the first swaps too."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    async def _rival() -> None:
        await embed_chunks(sessions, FakeEmbedder(), doc)

    await embed_chunks(sessions, FakeEmbedder(during=_rival), doc)

    assert await _chunk_texts(sessions, doc) == _expected_chunks(doc)


def _chunks(doc: uuid.UUID, tag: str) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=doc,
            chunk_index=i,
            text=f"{tag} {i}",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
            token_count=2,
        )
        for i in range(3)
    ]


async def _wait_for_a_lock_wait(sessions: Sessions) -> None:
    async with asyncio.timeout(10):
        while True:
            # A fresh transaction per poll: pg_stat_activity is a snapshot.
            async with sessions() as s:
                waiting = await s.scalar(
                    sa.text(
                        "SELECT count(*) FROM pg_stat_activity WHERE "
                        "wait_event_type = 'Lock' AND datname = current_database()"
                    )
                )
            if waiting:
                return
            await asyncio.sleep(0.05)


async def test_simultaneous_swaps_take_turns(sessions: Sessions) -> None:
    """A swap that starts while another is uncommitted waits, then replaces its set."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    async def _second() -> bool:
        async with sessions.begin() as s:
            return await replace_chunks(s, doc, _chunks(doc, "second"))

    async with sessions.begin() as first:
        assert await replace_chunks(first, doc, _chunks(doc, "first"))
        second = asyncio.create_task(_second())
        await _wait_for_a_lock_wait(sessions)
    assert await asyncio.wait_for(second, 10)

    assert await _chunk_texts(sessions, doc) == ["second 0", "second 1", "second 2"]


async def test_a_failure_write_waiting_on_a_swap_rechecks_and_changes_nothing(
    sessions: Sessions,
) -> None:
    """The guard holds under real concurrency, not only in sequence: the failure
    write waits on the swap's row lock, then re-checks the committed row's status."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    async def _losers_failure_write() -> bool:
        async with sessions.begin() as s:
            return await fail(s, EMBEDDING, doc, FailureReason.EMBEDDING_FAILED)

    async with sessions.begin() as winner:
        assert await replace_chunks(winner, doc, _chunks(doc, "winner"))
        loser = asyncio.create_task(_losers_failure_write())
        await _wait_for_a_lock_wait(sessions)
    assert await asyncio.wait_for(loser, 10) is False

    assert (await _row(sessions, doc))[:2] == (DocumentStatus.INDEXED, None)
    assert await _chunk_texts(sessions, doc) == ["winner 0", "winner 1", "winner 2"]


async def test_a_second_chunk_at_one_position_is_rejected(sessions: Sessions) -> None:
    """The database itself refuses a second chunk set (schema from the migrations)."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)
    async with sessions.begin() as s:
        s.add_all(_chunks(doc, "one"))

    with pytest.raises(sa.exc.IntegrityError, match="uq_chunks_document_chunk_index"):
        async with sessions.begin() as s:
            s.add(_chunks(doc, "two")[0])


async def test_a_swap_that_hits_a_database_error_is_transient(
    sessions: Sessions, test_engine: AsyncEngine
) -> None:
    """The swap touches the database: its failure is retried, never permanent, and
    never mistaken for a deleted document. A real error from PostgreSQL: the swap's
    row lock times out while another session holds it."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)
    impatient = create_async_engine(
        test_engine.url,
        poolclass=NullPool,
        connect_args={"server_settings": {"lock_timeout": "200ms"}},
    )
    try:
        async with test_engine.connect() as holder:
            await holder.begin()
            await holder.execute(
                sa.select(Document.id).where(Document.id == doc).with_for_update()
            )
            with pytest.raises(sa.exc.DBAPIError, match="lock timeout"):
                await embed_chunks(
                    async_sessionmaker(impatient, expire_on_commit=False),
                    FakeEmbedder(),
                    doc,
                )
            await holder.rollback()
    finally:
        await impatient.dispose()

    assert (await _row(sessions, doc)).status == DocumentStatus.READY
    assert await _chunk_texts(sessions, doc) == []


@pytest.mark.parametrize(
    "status",
    [
        DocumentStatus.PENDING,
        DocumentStatus.PROCESSING,
        DocumentStatus.INDEXED,
        DocumentStatus.FAILED,
    ],
)
async def test_a_document_no_longer_ready_is_not_embedded(
    sessions: Sessions, status: DocumentStatus
) -> None:
    doc = await _document(sessions, status=status, raw_text=TEXT)
    embedder = FakeEmbedder()

    await embed_chunks(sessions, embedder, doc)

    assert embedder.calls == 0
    assert (await _row(sessions, doc)).status == status


async def test_embedding_a_document_that_is_gone_does_nothing(
    sessions: Sessions,
) -> None:
    embedder = FakeEmbedder()
    await embed_chunks(sessions, embedder, uuid.uuid4())
    assert embedder.calls == 0


async def test_ready_without_text_is_an_internal_error(sessions: Sessions) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=None)

    with pytest.raises(PermanentFailure) as exc_info:
        await embed_chunks(sessions, FakeEmbedder(), doc)

    assert exc_info.value.reason == FailureReason.INTERNAL_ERROR


async def test_text_that_chunks_to_nothing_fails_permanently_as_no_text(
    sessions: Sessions,
) -> None:
    """D2/D3: a scanned PDF or whitespace has nothing to index; no embedding call."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=" \n\n \t ")
    embedder = FakeEmbedder()

    with pytest.raises(PermanentFailure) as exc_info:
        await embed_chunks(sessions, embedder, doc)

    assert exc_info.value.reason == FailureReason.NO_TEXT
    assert embedder.calls == 0


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
async def test_a_document_without_text_ends_as_no_text_end_to_end(
    sessions: Sessions, tmp_path: Path, content: bytes, content_type: ContentType
) -> None:
    """D2/D3 through both jobs: extraction succeeds (READY), there is nothing to
    index, and embedding fails it as no_text without calling the provider."""
    path = tmp_path / "blank"
    path.write_bytes(content)
    doc = await _document(sessions, file_path=str(path), content_type=content_type)
    embedder = FakeEmbedder()

    await _extract(sessions, doc)
    assert (await _row(sessions, doc)).status == DocumentStatus.READY
    with pytest.raises(PermanentFailure) as exc_info:
        await embed_chunks(sessions, embedder, doc)

    assert exc_info.value.reason == FailureReason.NO_TEXT
    assert embedder.calls == 0


async def test_a_chunking_failure_fails_permanently_without_embedding(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    def _broken_chunker(content_type: ContentType) -> object:
        raise ValueError("chunker bug")

    monkeypatch.setattr(jobs, "get_chunker", _broken_chunker)
    embedder = FakeEmbedder()

    with pytest.raises(PermanentFailure) as exc_info:
        await embed_chunks(sessions, embedder, doc)

    assert exc_info.value.reason == FailureReason.INTERNAL_ERROR
    assert embedder.calls == 0


async def test_an_embedding_failure_is_transient_and_writes_nothing(
    sessions: Sessions,
) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    with pytest.raises(EmbeddingError):
        await embed_chunks(sessions, FakeEmbedder(error=ConnectionError("503")), doc)

    assert (await _row(sessions, doc)).status == DocumentStatus.READY
    assert await _chunk_texts(sessions, doc) == []


async def test_a_document_deleted_during_embedding_gets_no_chunks(
    sessions: Sessions,
) -> None:
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    async def _delete() -> None:
        async with sessions.begin() as s:
            await s.execute(sa.delete(Document).where(Document.id == doc))

    await embed_chunks(sessions, FakeEmbedder(during=_delete), doc)

    assert await _chunk_texts(sessions, doc) == []


async def test_a_run_under_way_still_finishes_indexed_over_a_rivals_failed(
    sessions: Sessions,
) -> None:
    """A success beats a rival's failure; its reason goes with it (ADR-025)."""
    doc = await _document(sessions, status=DocumentStatus.READY, raw_text=TEXT)

    async def _rival_fails() -> None:
        async with sessions.begin() as s:
            await s.execute(
                sa.update(Document)
                .where(Document.id == doc)
                .values(
                    status=DocumentStatus.FAILED,
                    failure_reason=FailureReason.EMBEDDING_FAILED,
                )
            )

    await embed_chunks(sessions, FakeEmbedder(during=_rival_fails), doc)

    assert (await _row(sessions, doc))[:2] == (DocumentStatus.INDEXED, None)
