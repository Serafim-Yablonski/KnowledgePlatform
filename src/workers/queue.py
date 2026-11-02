"""The pipeline's work queue: workers claim owed documents from the documents table
(ADR-027). Nothing else carries the work: a document's status says what it is owed,
``available_at`` when it may be claimed next.

A claim leases a row: ``available_at`` moves to now + the lease, so no other claim
takes it while a worker holds it. A worker that dies writes no outcome; its lease
runs out and the next claim takes the row, counting the attempt. Attempts that all
end this way use up the phase's budget, and the claim that finds it used up fails
the document as interrupted instead of running it again.

The caller owns the transaction and commits the claim, and each outcome, on its own:
never around the work.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import sqlalchemy as sa
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import (
    FAIL_EMBEDDING,
    FAIL_EXTRACTION,
    START_EMBEDDING,
    START_EXTRACTION,
    DocumentStatus,
    FailureReason,
    Transition,
)
from src.models.document import Document
from src.workers.transitions import guarded_update

logger = structlog.get_logger(__name__)

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
    # The upload's trace context: the attempt's span starts under it.
    trace_context: dict[str, str] | None = None


def claim_statement(phase: Phase, *, lease: timedelta, max_attempts: int) -> sa.Update:
    """Lease the longest-waiting row this phase is owed, skipping rows other claims
    hold; one whose budget is used up is failed as interrupted instead."""
    candidate = (
        sa.select(Document.id)
        .where(
            Document.status.in_(phase.start.allowed_from),
            Document.available_at <= sa.func.now(),
        )
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
            Document.id, Document.attempts, Document.status, Document.trace_context
        )
        .execution_options(synchronize_session=False)
    )


async def claim(
    session: AsyncSession, phase: Phase, *, lease: timedelta, max_attempts: int
) -> Claim | None:
    """The next document to work on in ``phase``, leased; None if none is owed."""
    statement = claim_statement(phase, lease=lease, max_attempts=max_attempts)
    while (row := (await session.execute(statement)).first()) is not None:
        if row.status != DocumentStatus.FAILED:
            return Claim(row.id, row.attempts, row.trace_context)
        logger.error(
            "document status FAILED",
            document_id=str(row.id),
            phase=phase.name,
            failure_reason=FailureReason.INTERRUPTED,
            attempts=row.attempts,
        )
    return None


def _held(phase: Phase, document_id: uuid.UUID) -> sa.Update:
    """An update of the row while it is still where this phase's claim put it."""
    return (
        sa.update(Document)
        .where(Document.id == document_id, Document.status == phase.start.to)
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )


async def retry_later(
    session: AsyncSession, phase: Phase, document_id: uuid.UUID, delay: timedelta
) -> bool:
    """After a transient failure: claimable again once ``delay`` has passed. False:
    the document moved on (or is gone), nothing written."""
    moved = await session.execute(
        _held(phase, document_id).values(available_at=sa.func.now() + delay)
    )
    return moved.first() is not None


async def release(session: AsyncSession, phase: Phase, document_id: uuid.UUID) -> bool:
    """On a stop: claimable at once, and the cut-off attempt doesn't count (the stop
    was ours). False: the document moved on, nothing written."""
    moved = await session.execute(
        _held(phase, document_id).values(
            available_at=sa.func.now(),
            attempts=sa.func.greatest(Document.attempts - 1, 0),
        )
    )
    return moved.first() is not None


async def fail(
    session: AsyncSession,
    phase: Phase,
    document_id: uuid.UUID,
    reason: FailureReason,
) -> bool:
    """FAILED with ``reason``, unless the document moved past this phase."""
    moved = await session.execute(
        guarded_update(document_id, phase.fails)
        .values(failure_reason=reason)
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    return moved.first() is not None


def hand_off() -> dict[str, Any]:
    """Values for the write that finishes a phase: the next one may claim the
    document at once, with a fresh budget."""
    return {"attempts": 0, "available_at": sa.func.now()}


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
