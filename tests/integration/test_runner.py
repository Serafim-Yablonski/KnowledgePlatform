"""The worker loop (src/workers/runner.py) on a real PostgreSQL: claims, the outcome
of each attempt, the attempt timeout, a stop, and the attempt's trace.

A scratch database of its own, like test_work_queue.py: a claim takes the oldest owed
row of the whole table. The jobs are the real ones where the outcome depends on them
(extract_text with a missing file or a stand-in parse child, embed_chunks with a fake
embedder), stand-ins where only the runner is under test. Retries are fast because
the tests shrink the poll interval and the backoff, never by sleeping past them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
)
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import logfire
import pytest
import sqlalchemy as sa
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import NullPool
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.core.config import settings
from src.domain.documents import ContentType, DocumentStatus, FailureReason
from src.models.document import Document
from src.workers import jobs, runner
from src.workers.pipeline import PermanentFailure
from src.workers.runner import Work, Worker
from tests.support.database import migrate, scratch_database

Sessions = async_sessionmaker[AsyncSession]
FAST = {
    "poll_interval": 0.02,
    "backoff_base": timedelta(milliseconds=1),
    "shutdown_grace": 1.0,
}
# Long enough for any test's work; a test that hangs fails here instead.
DEADLINE = 20.0


@pytest.fixture(scope="module")
def runner_db(postgres_container: PostgresContainer) -> Generator[str]:
    with scratch_database(postgres_container, "runner") as url:
        migrate(url)
        yield url.replace("postgresql+psycopg", "postgresql+asyncpg")


@pytest.fixture
async def sessions(runner_db: str) -> AsyncGenerator[Sessions]:
    engine = create_async_engine(runner_db, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(sa.text("TRUNCATE documents, workspaces, users CASCADE"))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _document(
    sessions: Sessions,
    status: DocumentStatus = DocumentStatus.PENDING,
    *,
    content_type: ContentType = ContentType.PLAINTEXT,
    file_path: str = "/nonexistent",
    raw_text: str | None = None,
    trace_context: dict[str, str] | None = None,
) -> uuid.UUID:
    user, ws, doc = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with sessions.begin() as s:
        await s.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, :email, 'x')"
            ),
            {"id": user, "email": f"{user}@example.com"},
        )
        await s.execute(
            sa.text(
                "INSERT INTO workspaces (id, name, slug, created_by) "
                "VALUES (:id, 'W', :slug, :user)"
            ),
            {"id": ws, "slug": f"w-{ws.hex[:8]}", "user": user},
        )
        await s.execute(
            sa.insert(Document).values(
                id=doc,
                workspace_id=ws,
                title="T",
                content_type=content_type,
                file_path=file_path,
                file_size_bytes=1,
                uploaded_by=user,
                status=status,
                raw_text=raw_text,
                trace_context=trace_context,
            )
        )
    return doc


async def _row(sessions: Sessions, doc: uuid.UUID) -> sa.Row[Any]:
    async with sessions() as s:
        return (
            await s.execute(
                sa.select(
                    Document.status,
                    Document.failure_reason,
                    Document.attempts,
                    (Document.available_at <= sa.func.now()).label("claimable"),
                    Document.available_at - sa.func.now(),
                ).where(Document.id == doc)
            )
        ).one()


def _work(
    work: Work,
    job: Callable[[uuid.UUID], Awaitable[None]] | None = None,
    **changes: Any,
) -> Work:
    return dataclasses.replace(
        work, **({"job": job} if job is not None else {}), **{**FAST, **changes}
    )


def _extract_work(sessions: Sessions, **changes: Any) -> Work:
    return _work(runner.extract_work(sessions, settings), **changes)


def _embed_work(sessions: Sessions, embedder: jobs.Embedder, **changes: Any) -> Work:
    return _work(runner.embed_work(sessions, embedder, settings), **changes)


@asynccontextmanager
async def _running(
    sessions: Sessions | Any, work: Work
) -> AsyncIterator[asyncio.Event]:
    """A worker running in the background; stopped, and awaited, on exit."""
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(sessions, work).run(stop))
    try:
        yield stop
    finally:
        stop.set()
        await asyncio.wait_for(task, DEADLINE)


async def _until(check: Callable[[], Awaitable[bool]]) -> None:
    async with asyncio.timeout(DEADLINE):
        while not await check():
            await asyncio.sleep(0.02)


async def _until_status(
    sessions: Sessions, doc: uuid.UUID, status: DocumentStatus
) -> sa.Row[Any]:
    async def reached() -> bool:
        return (await _row(sessions, doc)).status == status

    await _until(reached)
    return await _row(sessions, doc)


class Counting:
    """A stand-in job: counts its runs, then does what ``body`` does."""

    def __init__(self, body: Callable[[uuid.UUID], Awaitable[None]]) -> None:
        self.runs = 0
        self._body = body

    async def __call__(self, document_id: uuid.UUID) -> None:
        self.runs += 1
        await self._body(document_id)


# ---------------------------------------------------------------------------
# How an attempt ends
# ---------------------------------------------------------------------------


async def test_a_plaintext_upload_is_extracted_and_handed_to_embedding(
    sessions: Sessions, tmp_path: Path
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello")
    doc = await _document(sessions, file_path=str(path))

    async with _running(sessions, _extract_work(sessions)):
        row = await _until_status(sessions, doc, DocumentStatus.READY)

    # A fresh budget for embedding, claimable at once.
    assert (row.attempts, row.claimable) == (0, True)


async def test_a_permanent_failure_fails_the_document_on_the_first_attempt(
    sessions: Sessions,
) -> None:
    async def unreadable(_: uuid.UUID) -> None:
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)

    job = Counting(unreadable)
    doc = await _document(sessions)

    async with _running(sessions, _extract_work(sessions, job=job)):
        row = await _until_status(sessions, doc, DocumentStatus.FAILED)

    assert (row.failure_reason, row.attempts, job.runs) == (
        FailureReason.UNREADABLE_FILE,
        1,
        1,
    )


async def test_a_file_that_stays_unreadable_fails_after_the_budget(
    sessions: Sessions,
) -> None:
    """A read error is transient: retried until the budget, then internal_error."""
    doc = await _document(sessions, file_path="/nonexistent/file.txt")

    async with _running(sessions, _extract_work(sessions)):
        row = await _until_status(sessions, doc, DocumentStatus.FAILED)

    assert (row.failure_reason, row.attempts) == (
        FailureReason.INTERNAL_ERROR,
        settings.PIPELINE_MAX_ATTEMPTS,
    )


async def test_a_parse_killed_on_every_attempt_fails_as_too_large(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child.py"
    child.write_text("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    monkeypatch.setattr(jobs, "_PDF_CHILD", child)
    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF")
    doc = await _document(sessions, content_type=ContentType.PDF, file_path=str(path))

    async with _running(sessions, _extract_work(sessions)):
        row = await _until_status(sessions, doc, DocumentStatus.FAILED)

    assert (row.failure_reason, row.attempts) == (
        FailureReason.EXTRACTION_TOO_LARGE,
        settings.PIPELINE_MAX_ATTEMPTS,
    )


class FailingEmbedder:
    """Fails the first ``failures`` calls, then embeds."""

    def __init__(self, failures: int) -> None:
        self.calls = 0
        self._failures = failures

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("503 from the provider")
        return [[0.1] * settings.EMBEDDING_DIMENSIONS for _ in texts]


async def test_an_embedding_that_keeps_failing_fails_as_embedding_failed(
    sessions: Sessions,
) -> None:
    embedder = FailingEmbedder(failures=10**6)
    doc = await _document(sessions, DocumentStatus.READY, raw_text="Some text here.")

    async with _running(sessions, _embed_work(sessions, embedder)):
        row = await _until_status(sessions, doc, DocumentStatus.FAILED)

    assert (row.failure_reason, row.attempts, embedder.calls) == (
        FailureReason.EMBEDDING_FAILED,
        settings.PIPELINE_MAX_ATTEMPTS,
        settings.PIPELINE_MAX_ATTEMPTS,
    )


async def test_an_embedding_that_fails_once_is_retried_and_indexes(
    sessions: Sessions,
) -> None:
    embedder = FailingEmbedder(failures=1)
    doc = await _document(sessions, DocumentStatus.READY, raw_text="Some text here.")

    async with _running(sessions, _embed_work(sessions, embedder)):
        await _until_status(sessions, doc, DocumentStatus.INDEXED)

    assert embedder.calls == 2


async def test_a_transient_failure_waits_out_a_backoff_within_its_bound(
    sessions: Sessions,
) -> None:
    async def flaky(_: uuid.UUID) -> None:
        raise RuntimeError("database hiccup")

    job = Counting(flaky)
    base = timedelta(hours=1)
    doc = await _document(sessions)

    async with _running(sessions, _extract_work(sessions, job=job, backoff_base=base)):

        async def failed_once() -> bool:
            return job.runs == 1 and not (await _row(sessions, doc)).claimable

        await _until(failed_once)

    row = await _row(sessions, doc)
    # Full jitter: uniform in [0, base) for a first retry; never past the lease.
    assert (row.status, row.attempts) == (DocumentStatus.PROCESSING, 1)
    assert timedelta(0) < row[4] <= base


async def test_a_failure_that_cannot_be_recorded_is_logged_and_left_for_the_lease(
    sessions: Sessions,
) -> None:
    async def unreadable(_: uuid.UUID) -> None:
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)

    doc = await _document(sessions)
    outcome_written = asyncio.Event()

    class DatabaseDownAfterTheClaim:
        """The claim commits; the outcome's transaction can't connect."""

        def __init__(self) -> None:
            self.claims = 0

        def begin(self) -> Any:
            self.claims += 1
            if self.claims == 1:
                return sessions.begin()
            outcome_written.set()
            raise OperationalError("UPDATE documents", None, Exception("lost"))

    with structlog.testing.capture_logs() as logs:
        async with _running(
            DatabaseDownAfterTheClaim(), _extract_work(sessions, job=unreadable)
        ):
            await asyncio.wait_for(outcome_written.wait(), DEADLINE)

    row = await _row(sessions, doc)
    # Not terminal yet: the lease returns it, and the next claim counts the attempt.
    assert (row.status, row.attempts, row.claimable) == (
        DocumentStatus.PROCESSING,
        1,
        False,
    )
    assert {
        "event": "could not record the outcome; the lease will return the document",
        "document_id": str(doc),
        "error_type": "OperationalError",
        # The driver's exception, not SQLAlchemy's text with the statement.
        "error": "lost",
        "log_level": "error",
    }.items() <= next(
        log for log in logs if log["event"].startswith("could not record")
    ).items()


async def test_a_claim_that_fails_is_retried_and_the_worker_keeps_going(
    sessions: Sessions, tmp_path: Path
) -> None:
    """A database restart costs a poll, never the worker."""
    path = tmp_path / "a.txt"
    path.write_text("hello")
    doc = await _document(sessions, file_path=str(path))

    class DatabaseDownAtFirst:
        def __init__(self) -> None:
            self.calls = 0

        def begin(self) -> Any:
            self.calls += 1
            if self.calls <= 2:
                raise OperationalError("UPDATE documents", None, Exception("down"))
            return sessions.begin()

    with structlog.testing.capture_logs() as logs:
        async with _running(DatabaseDownAtFirst(), _extract_work(sessions)):
            await _until_status(sessions, doc, DocumentStatus.READY)

    assert [log["event"] for log in logs].count("claim failed; retrying") == 2


async def test_a_failed_claim_logs_the_databases_reason_not_the_statement(
    runner_db: str,
) -> None:
    """A worker started before the migrations: every poll fails with SQLAlchemy's
    generic ProgrammingError, and only the driver's message says why. SQLAlchemy's
    own text appends the statement and its parameters; the log leaves them out."""
    unmigrated = create_async_engine(
        runner_db,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": "nowhere"}},
    )
    sessions = async_sessionmaker(unmigrated, expire_on_commit=False)
    try:
        with structlog.testing.capture_logs() as logs:
            async with _running(sessions, _extract_work(sessions)):

                async def failed() -> bool:
                    return any(e["event"] == "claim failed; retrying" for e in logs)

                await _until(failed)
    finally:
        await unmigrated.dispose()

    first = next(e for e in logs if e["event"] == "claim failed; retrying")
    assert first["error_type"] == "ProgrammingError"
    assert 'relation "documents" does not exist' in first["error"]
    assert "UPDATE" not in first["error"]


async def test_a_losing_runs_failure_never_overwrites_a_winners_indexed(
    sessions: Sessions,
) -> None:
    doc = await _document(sessions, DocumentStatus.READY, raw_text="text")

    async def loses_to_a_rival(document_id: uuid.UUID) -> None:
        async with sessions.begin() as s:
            await s.execute(
                sa.update(Document)
                .where(Document.id == document_id)
                .values(status=DocumentStatus.INDEXED)
            )
        raise PermanentFailure(FailureReason.NO_TEXT)

    with structlog.testing.capture_logs() as logs:
        async with _running(
            sessions, _embed_work(sessions, FailingEmbedder(0), job=loses_to_a_rival)
        ):

            async def logged() -> bool:
                return any(
                    log["event"].startswith("failure not recorded") for log in logs
                )

            await _until(logged)

    assert (await _row(sessions, doc)).status == DocumentStatus.INDEXED


# ---------------------------------------------------------------------------
# The attempt's timeout
# ---------------------------------------------------------------------------


async def _hang(_: uuid.UUID) -> None:
    await asyncio.sleep(3600)


async def test_an_extraction_past_its_timeout_fails_permanently(
    sessions: Sessions,
) -> None:
    job = Counting(_hang)
    doc = await _document(sessions)

    async with _running(sessions, _extract_work(sessions, job=job, timeout=0.2)):
        row = await _until_status(sessions, doc, DocumentStatus.FAILED)

    assert (row.failure_reason, job.runs) == (FailureReason.EXTRACTION_TIMEOUT, 1)


async def test_an_embedding_past_its_timeout_is_retried(sessions: Sessions) -> None:
    job = Counting(_hang)
    doc = await _document(sessions, DocumentStatus.READY, raw_text="text")

    async with _running(
        sessions,
        _embed_work(
            sessions,
            FailingEmbedder(0),
            job=job,
            timeout=0.2,
            backoff_base=timedelta(hours=1),
        ),
    ):

        async def timed_out_once() -> bool:
            return job.runs == 1 and not (await _row(sessions, doc)).claimable

        await _until(timed_out_once)

    row = await _row(sessions, doc)
    assert (row.status, row.failure_reason, row.attempts) == (
        DocumentStatus.READY,
        None,
        1,
    )


async def test_a_timeout_the_job_raises_itself_is_not_the_attempts(
    sessions: Sessions,
) -> None:
    """A socket's TimeoutError is transient: only the attempt's own timer means the
    extraction took too long."""

    async def socket_timeout(_: uuid.UUID) -> None:
        raise TimeoutError("read timed out")

    doc = await _document(sessions)

    async with _running(
        sessions,
        _extract_work(sessions, job=socket_timeout, backoff_base=timedelta(hours=1)),
    ):

        async def retried() -> bool:
            return not (await _row(sessions, doc)).claimable

        await _until(retried)

    assert (await _row(sessions, doc)).status == DocumentStatus.PROCESSING


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


async def test_a_worker_runs_as_many_attempts_at_once_as_it_has_slots(
    sessions: Sessions,
) -> None:
    release = asyncio.Event()
    started: list[uuid.UUID] = []

    async def blocked(document_id: uuid.UUID) -> None:
        started.append(document_id)
        await release.wait()

    docs = [await _document(sessions) for _ in range(3)]

    async with _running(sessions, _extract_work(sessions, job=blocked, concurrency=2)):

        async def two_started() -> bool:
            return len(started) == 2

        await _until(two_started)
        await asyncio.sleep(0.2)  # ten polls: a third slot would have claimed by now
        assert len(started) == 2
        (waiting,) = set(docs) - set(started)
        assert (await _row(sessions, waiting)).attempts == 0

        release.set()

        async def all_started() -> bool:
            return len(started) == 3

        await _until(all_started)


async def test_an_idle_worker_claims_once_per_poll_whatever_its_slots(
    sessions: Sessions,
) -> None:
    """One claimer fills every slot: an idle worker's claims don't grow with them.
    The jittered delay is at least half the interval, so a window allows at most
    window / (interval / 2) claims plus the first; a claim per slot would be
    ``slots`` times that."""

    class CountedTransactions:
        def __init__(self) -> None:
            self.opened = 0

        def begin(self) -> Any:
            self.opened += 1
            return sessions.begin()

    # Each claim here opens a connection (~25 ms): the poll must dwarf it.
    poll, window, slots = 0.2, 1.0, 8
    transactions = CountedTransactions()

    async with _running(
        transactions, _extract_work(sessions, concurrency=slots, poll_interval=poll)
    ):
        await asyncio.sleep(window)

    assert 1 <= transactions.opened <= window / (poll / 2) + 1


# ---------------------------------------------------------------------------
# A stop
# ---------------------------------------------------------------------------


async def test_a_stop_releases_an_attempt_still_running_after_the_grace(
    sessions: Sessions,
) -> None:
    job = Counting(_hang)
    doc = await _document(sessions)
    loop = asyncio.get_running_loop()

    async with _running(sessions, _extract_work(sessions, job=job)) as stop:

        async def claimed() -> bool:
            return job.runs == 1

        await _until(claimed)
        stopped_at = loop.time()
        stop.set()
    took = loop.time() - stopped_at

    # Claimable at once, and the cut-off attempt doesn't count: the stop was ours.
    row = await _row(sessions, doc)
    assert (row.status, row.attempts, row.claimable) == (
        DocumentStatus.PROCESSING,
        0,
        True,
    )
    assert FAST["shutdown_grace"] <= took < FAST["shutdown_grace"] + 2


async def test_an_attempt_that_ends_within_the_grace_records_its_outcome(
    sessions: Sessions,
) -> None:
    async def slow_then_unreadable(_: uuid.UUID) -> None:
        await asyncio.sleep(0.3)
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)

    job = Counting(slow_then_unreadable)
    doc = await _document(sessions)

    async with _running(sessions, _extract_work(sessions, job=job)) as stop:

        async def claimed() -> bool:
            return job.runs == 1

        await _until(claimed)
        stop.set()

    row = await _row(sessions, doc)
    assert (row.status, row.failure_reason) == (
        DocumentStatus.FAILED,
        FailureReason.UNREADABLE_FILE,
    )


async def test_an_outcome_write_under_way_when_the_grace_ends_is_not_cut_off(
    sessions: Sessions,
) -> None:
    """Only jobs are cancelled: a lost write would leave the document to its lease,
    a cost the stop must not add."""
    grace = FAST["shutdown_grace"]
    doc = await _document(sessions)
    stopped = asyncio.Event()

    class SlowAfterTheStop:
        """Writes begun after the stop take longer than the grace period."""

        @asynccontextmanager
        async def begin(self) -> AsyncIterator[AsyncSession]:
            if stopped.is_set():
                await asyncio.sleep(grace + 0.5)
            async with sessions.begin() as session:
                yield session

    async def fails_once_stopped(_: uuid.UUID) -> None:
        await stopped.wait()
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)

    job = Counting(fails_once_stopped)
    async with _running(SlowAfterTheStop(), _extract_work(sessions, job=job)) as stop:

        async def claimed() -> bool:
            return job.runs == 1

        await _until(claimed)
        stopped.set()
        stop.set()

    assert (await _row(sessions, doc)).status == DocumentStatus.FAILED


async def test_a_stopped_worker_claims_nothing_more(sessions: Sessions) -> None:
    job = Counting(_hang)

    async with _running(sessions, _extract_work(sessions, job=job)) as stop:
        stop.set()
        doc = await _document(sessions)
        await asyncio.sleep(0.2)

    assert job.runs == 0
    assert (await _row(sessions, doc)).attempts == 0


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


@pytest.fixture
def spans() -> Generator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]
    try:
        yield exporter
    finally:
        processor.shutdown()


def _named(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


async def test_an_attempt_continues_the_uploads_trace(
    sessions: Sessions, spans: InMemorySpanExporter
) -> None:
    """One trace from the upload to the outcome (ADR-027): the attempt is a child of
    the span the upload ran in, and the job's spans are the attempt's."""
    with logfire.span("upload") as upload:
        carrier = dict(logfire.get_context())
    doc = await _document(sessions, trace_context=carrier)

    async def traced_job(_: uuid.UUID) -> None:
        with logfire.span("job work"):
            pass

    async with _running(sessions, _extract_work(sessions, job=traced_job)):

        async def finished() -> bool:
            return any(s.name == "job work" for s in spans.get_finished_spans())

        await _until(finished)

    attempt = _named(spans, "extract attempt")
    work = _named(spans, "job work")
    uploaded = upload.get_span_context()
    assert attempt.context is not None and attempt.parent is not None
    assert attempt.context.trace_id == uploaded.trace_id
    assert attempt.parent.span_id == uploaded.span_id
    assert work.parent is not None and work.parent.span_id == attempt.context.span_id
    assert (attempt.attributes or {})["document_id"] == str(doc)


async def test_an_attempt_without_a_stored_context_starts_its_own_trace(
    sessions: Sessions, spans: InMemorySpanExporter
) -> None:
    await _document(sessions)

    async def done(_: uuid.UUID) -> None:
        pass

    async with _running(sessions, _extract_work(sessions, job=done)):

        async def finished() -> bool:
            return any(s.name == "extract attempt" for s in spans.get_finished_spans())

        await _until(finished)

    assert _named(spans, "extract attempt").parent is None


async def test_a_failure_as_interrupted_ends_the_uploads_trace(
    sessions: Sessions, spans: InMemorySpanExporter
) -> None:
    """A crash loop's attempts died unrecorded, and their spans with them; the claim
    that fails the document records it under the upload's trace, so the trace says
    how the upload ended."""
    with logfire.span("upload") as upload:
        carrier = dict(logfire.get_context())
    doc = await _document(sessions, DocumentStatus.PROCESSING, trace_context=carrier)
    async with sessions.begin() as s:
        await s.execute(
            sa.update(Document)
            .where(Document.id == doc)
            .values(attempts=settings.PIPELINE_MAX_ATTEMPTS)
        )
    job = Counting(_hang)

    with structlog.testing.capture_logs() as logs:
        async with _running(sessions, _extract_work(sessions, job=job)):

            async def recorded() -> bool:
                names = {s.name for s in spans.get_finished_spans()}
                return "extract interrupted" in names

            await _until(recorded)

    interrupted = _named(spans, "extract interrupted")
    uploaded = upload.get_span_context()
    assert interrupted.context is not None and interrupted.parent is not None
    assert interrupted.context.trace_id == uploaded.trace_id
    assert interrupted.parent.span_id == uploaded.span_id
    assert {
        k: (interrupted.attributes or {})[k]
        for k in ("document_id", "failure_reason", "attempts")
    } == {
        "document_id": str(doc),
        "failure_reason": "interrupted",
        "attempts": settings.PIPELINE_MAX_ATTEMPTS,
    }
    assert job.runs == 0
    failed = [e for e in logs if e["event"] == "document status FAILED"]
    assert [e["document_id"] for e in failed] == [str(doc)]
