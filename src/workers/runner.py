"""The pipeline's worker loop (ADR-027): claim owed documents, run each attempt, write
its outcome.

One claimer fills N slots. It claims while a slot is free and sleeps about a poll
interval when nothing is owed, so an idle worker runs one claim query per interval
whatever N is. Each claim and each outcome is a short transaction of its own.

How an attempt ends decides its outcome. The job returns: it wrote its own result.
It raises PermanentFailure: FAILED with that reason. Anything else is transient: the
document waits out a backoff, or fails with the phase's reason once the budget is
used up. An attempt that ends unrecorded (the worker killed, the database gone) is
the lease's: it runs out and the next claim counts the attempt.

A stop ends claiming, gives running attempts a grace period, then cancels their jobs
and releases their documents. Outcome and release writes are never cancelled.
"""

from __future__ import annotations

import asyncio
import random
import signal
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Protocol

import logfire
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import Settings
from src.domain.documents import FailureReason
from src.workers import jobs, queue
from src.workers.pipeline import PermanentFailure, failure_reason
from src.workers.queue import EMBEDDING, EXTRACTION, Claim, Phase

logger = structlog.get_logger(__name__)


class Transactions(Protocol):
    """Where the runner's own writes go: ``begin()`` opens one transaction."""

    def begin(self) -> Any: ...


@dataclass(frozen=True)
class Work:
    """What one worker process runs: a phase, its job and its limits."""

    phase: Phase
    job: Callable[[uuid.UUID], Awaitable[None]]
    concurrency: int
    # Seconds an attempt may run. Shorter than the lease: a live attempt never loses
    # its document (settings validate it).
    timeout: float
    lease: timedelta
    # What an attempt that runs out of time records; None: transient.
    timed_out: FailureReason | None
    backoff_base: timedelta
    max_attempts: int
    poll_interval: float
    shutdown_grace: float
    backoff_cap: timedelta = timedelta(minutes=10)


def extract_work(sessions: async_sessionmaker[AsyncSession], cfg: Settings) -> Work:
    return Work(
        phase=EXTRACTION,
        job=partial(
            jobs.extract_text,
            sessions,
            parse_timeout=cfg.PIPELINE_EXTRACT_PARSE_TIMEOUT,
        ),
        concurrency=cfg.PIPELINE_EXTRACT_CONCURRENCY,
        timeout=cfg.PIPELINE_EXTRACT_TIMEOUT,
        lease=timedelta(seconds=cfg.PIPELINE_EXTRACT_LEASE),
        # The parse child's own timeout comes first; this is the backstop, and the
        # same file would run out of time again.
        timed_out=FailureReason.EXTRACTION_TIMEOUT,
        backoff_base=timedelta(seconds=cfg.PIPELINE_EXTRACT_BACKOFF_BASE),
        max_attempts=cfg.PIPELINE_MAX_ATTEMPTS,
        poll_interval=cfg.PIPELINE_POLL_INTERVAL,
        shutdown_grace=cfg.PIPELINE_SHUTDOWN_GRACE,
    )


def embed_work(
    sessions: async_sessionmaker[AsyncSession], embedder: jobs.Embedder, cfg: Settings
) -> Work:
    return Work(
        phase=EMBEDDING,
        job=partial(jobs.embed_chunks, sessions, embedder),
        concurrency=cfg.PIPELINE_EMBED_CONCURRENCY,
        timeout=cfg.PIPELINE_EMBED_TIMEOUT,
        lease=timedelta(seconds=cfg.PIPELINE_EMBED_LEASE),
        # Nearly all of it is the provider's time: a slow spell, worth another try.
        timed_out=None,
        backoff_base=timedelta(seconds=cfg.PIPELINE_EMBED_BACKOFF_BASE),
        max_attempts=cfg.PIPELINE_MAX_ATTEMPTS,
        poll_interval=cfg.PIPELINE_POLL_INTERVAL,
        shutdown_grace=cfg.PIPELINE_SHUTDOWN_GRACE,
    )


class Worker:
    def __init__(self, sessions: Transactions, work: Work) -> None:
        self._sessions = sessions
        self._work = work
        self._attempts: set[asyncio.Task[None]] = set()
        self._jobs: set[asyncio.Task[None]] = set()

    async def run(self, stop: asyncio.Event) -> None:
        """Claim and run attempts until ``stop`` is set, then wind down."""
        stopping = asyncio.create_task(stop.wait())
        try:
            while not stop.is_set():
                if len(self._attempts) >= self._work.concurrency:
                    await asyncio.wait(
                        {*self._attempts, stopping},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    continue
                claim = await self._claim()
                if claim is None:
                    await asyncio.wait({stopping}, timeout=self._poll_delay())
                    continue
                attempt = asyncio.create_task(self._attempt(claim))
                self._attempts.add(attempt)
                attempt.add_done_callback(self._attempts.discard)
        finally:
            stopping.cancel()
            await self._wind_down()

    def _poll_delay(self) -> float:
        # Jittered, so workers started together don't poll in step.
        return self._work.poll_interval * random.uniform(0.5, 1.5)

    async def _claim(self) -> Claim | None:
        try:
            # An idle worker claims every interval: no trace per empty poll.
            with logfire.suppress_instrumentation():
                async with self._sessions.begin() as session:
                    return await queue.claim(
                        session,
                        self._work.phase,
                        lease=self._work.lease,
                        max_attempts=self._work.max_attempts,
                    )
        except Exception as exc:  # noqa: BLE001 — a restarting database costs a poll
            logger.warning(
                "claim failed; retrying",
                phase=self._work.phase.name,
                error_type=type(exc).__name__,
            )
            return None

    async def _wind_down(self) -> None:
        """Running attempts get the grace period; then their jobs are cancelled, and
        each attempt releases its document."""
        if not self._attempts:
            return
        await asyncio.wait(set(self._attempts), timeout=self._work.shutdown_grace)
        for job in set(self._jobs):
            job.cancel()
        await asyncio.gather(*self._attempts, return_exceptions=True)

    async def _attempt(self, claim: Claim) -> None:
        document_id = claim.document_id
        # The upload's trace continues here (ADR-027); a row without one starts its own.
        with (
            logfire.attach_context(claim.trace_context or {}),
            logfire.span(
                "{phase} attempt",
                _span_name=f"{self._work.phase.name} attempt",
                phase=self._work.phase.name,
                document_id=str(document_id),
                attempt=claim.attempt,
            ) as span,
        ):
            # A task of its own, so a stop can cancel the job and nothing else.
            job = asyncio.create_task(self._run_job(document_id))
            self._jobs.add(job)
            job.add_done_callback(self._jobs.discard)
            try:
                await job
            except asyncio.CancelledError:
                if not job.cancelled():
                    raise
                span.set_attribute("outcome", "released")
                await self._record(document_id, queue.release, self._work.phase)
                logger.info("released on stop", document_id=str(document_id))
            except PermanentFailure as exc:
                span.set_attribute("outcome", "failed")
                await self._fail(claim, exc.reason, exc)
            except Exception as exc:  # noqa: BLE001 — any other failure is transient
                await self._transient(claim, exc, span)
            else:
                span.set_attribute("outcome", "done")

    async def _run_job(self, document_id: uuid.UUID) -> None:
        deadline = asyncio.timeout(self._work.timeout)
        try:
            async with deadline:
                await self._work.job(document_id)
        except TimeoutError as exc:
            # Only the attempt's own timer: a TimeoutError the job raised (a socket's,
            # say) is an ordinary transient failure.
            if deadline.expired() and self._work.timed_out is not None:
                raise PermanentFailure(self._work.timed_out) from exc
            raise

    async def _transient(
        self, claim: Claim, exc: Exception, span: logfire.LogfireSpan
    ) -> None:
        if claim.attempt >= self._work.max_attempts:
            span.set_attribute("outcome", "failed")
            await self._fail(claim, failure_reason(exc), exc)
            return
        span.set_attribute("outcome", "retry")
        delay = queue.backoff(
            claim.attempt, self._work.backoff_base, self._work.backoff_cap
        )
        logger.warning(
            "attempt failed; retrying later",
            document_id=str(claim.document_id),
            phase=self._work.phase.name,
            attempt=claim.attempt,
            error_type=type(exc).__name__,
            retry_in_s=round(delay.total_seconds(), 1),
        )
        await self._record(
            claim.document_id, queue.retry_later, self._work.phase, delay=delay
        )

    async def _fail(self, claim: Claim, reason: FailureReason, exc: Exception) -> None:
        moved = await self._record(
            claim.document_id, queue.fail, self._work.phase, reason=reason
        )
        if moved is None:
            return
        if not moved:
            logger.info(
                "failure not recorded: the document has moved past this stage",
                document_id=str(claim.document_id),
                failure_reason=reason,
            )
            return
        logger.error(
            "document status FAILED",
            document_id=str(claim.document_id),
            phase=self._work.phase.name,
            failure_reason=reason,
            attempts=claim.attempt,
            error_type=type(exc).__name__,
        )

    async def _record(
        self,
        document_id: uuid.UUID,
        write: Callable[..., Awaitable[bool]],
        phase: Phase,
        **values: Any,
    ) -> bool | None:
        """One outcome write in its own transaction. None: it could not be made, and
        the document waits for its lease."""
        try:
            async with self._sessions.begin() as session:
                moved: bool = await write(session, phase, document_id, **values)
                return moved
        except Exception as exc:  # noqa: BLE001 — never lose the worker over one write
            logger.error(
                "could not record the outcome; the lease will return the document",
                document_id=str(document_id),
                error_type=type(exc).__name__,
            )
            return None


async def serve(
    make_work: Callable[
        [async_sessionmaker[AsyncSession], AsyncExitStack], Awaitable[Work]
    ],
    database_url: str,
) -> None:
    """Run one worker until SIGTERM or SIGINT, on an engine and clients of its own.

    ``make_work`` builds the work; clients it opens go on the stack, closed after the
    worker has wound down.
    """
    engine = create_async_engine(database_url, pool_pre_ping=True)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(engine.dispose)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        work = await make_work(sessions, stack)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        logger.info(
            "worker started", phase=work.phase.name, concurrency=work.concurrency
        )
        await Worker(sessions, work).run(stop)
        logger.info("worker stopped", phase=work.phase.name)
