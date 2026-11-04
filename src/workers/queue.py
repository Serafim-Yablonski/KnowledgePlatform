"""The pipeline's work queue: workers claim owed documents from the documents table
(ADR-027). Nothing else carries the work: a document's status says what it is owed,
``available_at`` when it may be claimed next.

A claim leases a row: ``available_at`` moves to now + the lease, so no other claim
takes it while a worker holds it. A worker that dies writes no outcome; its lease
runs out and the next claim takes the row, counting the attempt. Attempts that all
end this way use up the phase's budget, and the claim that finds it used up fails
the document as interrupted instead of running it again.

A lease is the claim's: only the worker holding it may move it. ``retry_later`` and
``release`` apply only while the row still carries the claim's lease deadline, so a
worker whose lease ran out and was claimed again can't free or shorten the new
holder's. Outcomes are the document's, not the lease's: ``fail`` and the jobs' own
writes are guarded by status alone (ADR-025), whichever run gets there.

The caller owns the transaction and commits the claim, and each outcome, on its own:
never around the work.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import (
    FAIL_EMBEDDING,
    FAIL_EXTRACTION,
    REINDEX,
    START_EMBEDDING,
    START_EXTRACTION,
    DocumentStatus,
    FailureReason,
    Transition,
)
from src.models.document import Document
from src.workers.transitions import guarded_update

_status = Document.__table__.c.status.type
_reason = Document.__table__.c.failure_reason.type


@dataclass(frozen=True)
class Phase:
    """A stage of the pipeline that workers claim documents for."""

    name: str
    # The claim's status write; its from-statuses are the work this phase is owed.
    start: Transition
    fails: Transition


EXTRACTION = Phase("extract", START_EXTRACTION, FAIL_EXTRACTION)
EMBEDDING = Phase("embed", START_EMBEDDING, FAIL_EMBEDDING)


@dataclass(frozen=True)
class Claim:
    document_id: uuid.UUID
    # 1 for the phase's first attempt.
    attempt: int
    # The lease's deadline, as the claim wrote it: the claim's fencing token.
    # ``attempts`` is no token: a hand-off or a re-index resets it, so a stalled run
    # can share its number with the claim that replaced it.
    lease_ends: datetime
    # The upload's trace context: the attempt's span starts under it.
    trace_context: dict[str, str] | None = None


@dataclass(frozen=True)
class Interrupted:
    """A row the claim failed as interrupted: its budget was used up by attempts
    that ended unrecorded."""

    document_id: uuid.UUID
    attempts: int
    trace_context: dict[str, str] | None


@dataclass(frozen=True)
class Claimed:
    """What one claim transaction did: the document leased, if any, and the rows it
    failed on the way. The caller reports those once the transaction commits."""

    claim: Claim | None
    interrupted: tuple[Interrupted, ...] = ()


def due(phase: Phase) -> sa.ColumnElement[bool]:
    """A row this phase is owed and may claim now: not leased, not backing off."""
    return sa.and_(
        Document.status.in_(phase.start.allowed_from),
        Document.available_at <= sa.func.now(),
    )


def lag_statement(phase: Phase) -> sa.Select[tuple[float]]:
    """Seconds the longest-waiting due row has waited for a claim; 0 when none is
    due. Rows a claim holds or a backoff hides don't count: nobody owes them a
    claim yet."""
    waited = sa.func.extract(
        "epoch", sa.func.now() - sa.func.min(Document.available_at)
    )
    return sa.select(sa.cast(sa.func.coalesce(waited, 0), sa.Float)).where(due(phase))


def claim_statement(phase: Phase, *, lease: timedelta, max_attempts: int) -> sa.Update:
    """Lease the longest-waiting row this phase is owed, skipping rows other claims
    hold; one whose budget is used up is failed as interrupted instead."""
    candidate = (
        sa.select(Document.id)
        .where(due(phase))
        .order_by(Document.available_at)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    used_up = Document.attempts >= max_attempts
    return (
        sa.update(Document)
        .where(Document.id == candidate)
        .values(
            status=sa.case(
                (used_up, sa.literal(DocumentStatus.FAILED, _status)),
                else_=sa.literal(phase.start.to, _status),
            ),
            failure_reason=sa.case(
                (used_up, sa.literal(FailureReason.INTERRUPTED, _reason)),
                else_=sa.null(),
            ),
            attempts=sa.case((used_up, Document.attempts), else_=Document.attempts + 1),
            available_at=sa.func.now() + lease,
        )
        .returning(
            Document.id,
            Document.attempts,
            Document.status,
            Document.available_at,
            Document.trace_context,
        )
        .execution_options(synchronize_session=False)
    )


async def claim(
    session: AsyncSession, phase: Phase, *, lease: timedelta, max_attempts: int
) -> Claimed:
    """The next document to work on in ``phase``, leased (none if none is owed), and
    the rows failed as interrupted before it."""
    statement = claim_statement(phase, lease=lease, max_attempts=max_attempts)
    interrupted: list[Interrupted] = []
    while (row := (await session.execute(statement)).first()) is not None:
        if row.status != DocumentStatus.FAILED:
            leased = Claim(row.id, row.attempts, row.available_at, row.trace_context)
            return Claimed(leased, tuple(interrupted))
        interrupted.append(Interrupted(row.id, row.attempts, row.trace_context))
    return Claimed(None, tuple(interrupted))


def _held(phase: Phase, claim: Claim) -> sa.Update:
    """An update of the row while ``claim`` still holds its lease."""
    return (
        sa.update(Document)
        .where(
            Document.id == claim.document_id,
            Document.status == phase.start.to,
            Document.available_at == claim.lease_ends,
        )
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )


async def retry_later(
    session: AsyncSession, phase: Phase, claim: Claim, delay: timedelta
) -> bool:
    """After a transient failure: claimable again once ``delay`` has passed. False:
    the lease is no longer this claim's, nothing written."""
    moved = await session.execute(
        _held(phase, claim).values(available_at=sa.func.now() + delay)
    )
    return moved.first() is not None


async def release(session: AsyncSession, phase: Phase, claim: Claim) -> bool:
    """On a stop: claimable at once, and the cut-off attempt doesn't count (the stop
    was ours). False: the lease is no longer this claim's, nothing written."""
    moved = await session.execute(
        _held(phase, claim).values(
            available_at=sa.func.now(),
            attempts=sa.func.greatest(Document.attempts - 1, 0),
        )
    )
    return moved.first() is not None


async def fail(
    session: AsyncSession,
    phase: Phase,
    claim: Claim,
    reason: FailureReason,
) -> bool:
    """FAILED with ``reason``, unless the document moved past this phase; not fenced
    by the lease (see the module docstring)."""
    moved = await session.execute(
        guarded_update(claim.document_id, phase.fails)
        .values(failure_reason=reason)
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    return moved.first() is not None


def hand_off() -> dict[str, Any]:
    """Values for the write that finishes a phase: the next one may claim the
    document at once, with a fresh budget."""
    return {"attempts": 0, "available_at": sa.func.now()}


async def reindex(session: AsyncSession, trace_context: dict[str, str] | None) -> int:
    """Owe embedding again to every document whose text was extracted (ADR-030):
    READY, claimable at once with a fresh budget, its attempts traced under
    ``trace_context``. Returns how many. Documents already owed work keep their
    place, so a claim holding one keeps its lease; FAILED ones without text never
    reached embedding and need a new upload."""
    owed = await session.execute(
        sa.update(Document)
        .where(
            Document.status.in_(REINDEX.allowed_from),
            Document.raw_text.is_not(None),
        )
        .values(
            status=REINDEX.to,
            failure_reason=None,
            trace_context=trace_context,
            **hand_off(),
        )
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    return len(owed.all())


def backoff(
    attempt: int,
    base: timedelta,
    cap: timedelta,
    rng: Callable[[], float] = random.random,
) -> timedelta:
    """Full jitter: uniform in [0, min(cap, base · 2^(attempt-1))). Spreads retries
    so documents that failed together don't all come back together."""
    ceiling: timedelta = min(cap, base * 2 ** (attempt - 1))
    return ceiling * rng()
