"""The documents table as the pipeline's work queue (ADR-027), on a real PostgreSQL.

A scratch database of its own: a claim takes the oldest owed row of the whole table,
so rows other tests leave behind would be claimed here. Tests move time by writing
``available_at``, never by sleeping.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy import NullPool
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.domain.documents import FINISH_EXTRACTION, DocumentStatus, FailureReason
from src.models.document import Document
from src.workers.queue import (
    EMBEDDING,
    EXTRACTION,
    Claim,
    Phase,
    backoff,
    claim,
    claim_statement,
    fail,
    hand_off,
    release,
    retry_later,
)
from src.workers.transitions import guarded_update
from tests.support.database import migrate, scratch_database

LEASE = timedelta(minutes=5)
BUDGET = 5
# Long enough for a claim that should not wait; a blocked claim fails the test.
NO_WAIT = 5.0


@pytest.fixture(scope="module")
def queue_db(postgres_container: PostgresContainer) -> Generator[str]:
    with scratch_database(postgres_container, "queue") as url:
        migrate(url)
        yield url.replace("postgresql+psycopg", "postgresql+asyncpg")


@pytest.fixture
async def sessions(queue_db: str) -> AsyncGenerator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(queue_db, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(sa.text("TRUNCATE documents, workspaces, users CASCADE"))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _documents(
    sessions: async_sessionmaker[AsyncSession],
    *statuses: DocumentStatus,
    waited: list[timedelta] | None = None,
) -> list[uuid.UUID]:
    """One document per status, in one workspace; ``waited[i]``: how long the i-th
    has been claimable (default: all since the same instant)."""
    user, ws = uuid.uuid4(), uuid.uuid4()
    ids = [uuid.uuid4() for _ in statuses]
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
        for i, (doc, status) in enumerate(zip(ids, statuses, strict=True)):
            await s.execute(
                sa.insert(Document).values(
                    id=doc,
                    workspace_id=ws,
                    title="T",
                    content_type="PDF",
                    file_path="/x.pdf",
                    file_size_bytes=1,
                    uploaded_by=user,
                    status=status,
                    available_at=sa.func.now() - (waited[i] if waited else timedelta()),
                )
            )
    return ids


async def _row(
    sessions: async_sessionmaker[AsyncSession], doc: uuid.UUID
) -> sa.Row[tuple[DocumentStatus, FailureReason | None, int, bool]]:
    async with sessions() as s:
        return (
            await s.execute(
                sa.select(
                    Document.status,
                    Document.failure_reason,
                    Document.attempts,
                    (Document.available_at > sa.func.now()).label("hidden"),
                ).where(Document.id == doc)
            )
        ).one()


async def _claim(
    sessions: async_sessionmaker[AsyncSession],
    phase: Phase = EXTRACTION,
    *,
    plan: str | None = None,
) -> Claim | None:
    """One claim in its own transaction; ``plan``: a SET LOCAL run first."""
    async with sessions.begin() as s:
        if plan is not None:
            await s.execute(sa.text(f"SET LOCAL {plan}"))
        return await asyncio.wait_for(
            claim(s, phase, lease=LEASE, max_attempts=BUDGET), NO_WAIT
        )


async def _expire(sessions: async_sessionmaker[AsyncSession], doc: uuid.UUID) -> None:
    async with sessions.begin() as s:
        await s.execute(
            sa.update(Document)
            .where(Document.id == doc)
            .values(available_at=sa.func.now() - timedelta(seconds=1))
        )


async def test_a_claim_leases_the_document_and_counts_the_attempt(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.PENDING)

    claimed = await _claim(sessions)

    assert claimed is not None
    assert (claimed.document_id, claimed.attempt) == (doc, 1)
    assert tuple(await _row(sessions, doc)) == (
        DocumentStatus.PROCESSING,
        None,
        1,
        True,
    )
    # Leased: no second worker gets it while the first holds it.
    assert await _claim(sessions) is None


async def test_a_claim_carries_the_uploads_trace_context(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The attempt continues the trace of the upload that made the row owed."""
    carrier = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    traced, untraced = await _documents(
        sessions,
        DocumentStatus.PENDING,
        DocumentStatus.PENDING,
        waited=[timedelta(minutes=2), timedelta(minutes=1)],
    )
    async with sessions.begin() as s:
        await s.execute(
            sa.update(Document)
            .where(Document.id == traced)
            .values(trace_context=carrier)
        )

    first, second = await _claim(sessions), await _claim(sessions)

    assert first is not None and second is not None
    assert (first.document_id, first.trace_context) == (traced, carrier)
    assert (second.document_id, second.trace_context) == (untraced, None)


async def test_a_claim_skips_a_row_another_claim_holds_instead_of_waiting(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    older, newer = await _documents(
        sessions,
        DocumentStatus.PENDING,
        DocumentStatus.PENDING,
        waited=[timedelta(minutes=2), timedelta(minutes=1)],
    )
    async with sessions.begin() as holder:
        held = await claim(holder, EXTRACTION, lease=LEASE, max_attempts=BUDGET)
        # The holder hasn't committed: its row is locked, and it is the oldest.
        rival = await _claim(sessions)
        third = await _claim(sessions)

    assert held is not None and held.document_id == older
    assert rival is not None and rival.document_id == newer
    assert third is None


async def test_an_expired_lease_is_claimed_again_and_counted(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.PENDING)
    await _claim(sessions)
    # The worker died: nobody wrote an outcome, the lease ran out.
    await _expire(sessions, doc)

    claimed = await _claim(sessions)

    assert claimed is not None
    assert (claimed.document_id, claimed.attempt) == (doc, 2)


@pytest.mark.parametrize(
    ("phase", "owed"),
    [
        (EXTRACTION, {DocumentStatus.PENDING, DocumentStatus.PROCESSING}),
        (EMBEDDING, {DocumentStatus.READY}),
    ],
)
async def test_a_claim_serves_only_its_phase(
    sessions: async_sessionmaker[AsyncSession],
    phase: Phase,
    owed: set[DocumentStatus],
) -> None:
    statuses = list(DocumentStatus)
    ids = dict(zip(statuses, await _documents(sessions, *statuses), strict=True))

    claimed = set()
    while (c := await _claim(sessions, phase)) is not None:
        claimed.add(c.document_id)

    assert claimed == {ids[s] for s in owed}


async def test_the_longest_waiting_document_is_claimed_first(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    newest, oldest, middle = await _documents(
        sessions,
        *[DocumentStatus.PENDING] * 3,
        waited=[timedelta(minutes=1), timedelta(minutes=3), timedelta(minutes=2)],
    )

    # Inserted newest first: a scan in table order would return it first. Index
    # scans off: the order must come from ORDER BY, whatever plan is chosen.
    order = []
    for _ in range(3):
        claimed = await _claim(sessions, plan="enable_indexscan = off")
        assert claimed is not None
        order.append(claimed.document_id)

    assert order == [oldest, middle, newest]


async def test_a_used_up_budget_fails_the_document_as_interrupted(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    worn, fresh = await _documents(
        sessions,
        DocumentStatus.PROCESSING,
        DocumentStatus.PENDING,
        waited=[timedelta(minutes=2), timedelta(minutes=1)],
    )
    async with sessions.begin() as s:
        await s.execute(
            sa.update(Document).where(Document.id == worn).values(attempts=BUDGET)
        )

    # Every attempt ended unrecorded: no more runs; the claim moves on to the next.
    claimed = await _claim(sessions)

    assert claimed is not None and claimed.document_id == fresh
    assert tuple(await _row(sessions, worn))[:3] == (
        DocumentStatus.FAILED,
        FailureReason.INTERRUPTED,
        BUDGET,
    )


async def test_retry_later_hides_the_document_until_its_delay_passes(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.PENDING)
    await _claim(sessions)

    async with sessions.begin() as s:
        assert await retry_later(s, EXTRACTION, doc, timedelta(minutes=10))

    assert tuple(await _row(sessions, doc)) == (
        DocumentStatus.PROCESSING,
        None,
        1,
        True,
    )
    assert await _claim(sessions) is None
    await _expire(sessions, doc)
    claimed = await _claim(sessions)
    assert claimed is not None and claimed.attempt == 2


async def test_release_makes_the_document_claimable_at_once_and_refunds_the_attempt(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.PENDING)
    await _claim(sessions)

    async with sessions.begin() as s:
        assert await release(s, EXTRACTION, doc)

    # The stop was ours, not the document's.
    claimed = await _claim(sessions)
    assert claimed is not None and (claimed.document_id, claimed.attempt) == (doc, 1)


async def test_fail_records_the_reason(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.READY)
    await _claim(sessions, EMBEDDING)

    async with sessions.begin() as s:
        assert await fail(s, EMBEDDING, doc, FailureReason.EMBEDDING_FAILED)

    assert tuple(await _row(sessions, doc))[:2] == (
        DocumentStatus.FAILED,
        FailureReason.EMBEDDING_FAILED,
    )


@pytest.mark.parametrize("moved_to", [DocumentStatus.INDEXED, DocumentStatus.FAILED])
async def test_outcome_writes_leave_a_document_that_moved_on_alone(
    sessions: async_sessionmaker[AsyncSession], moved_to: DocumentStatus
) -> None:
    """A worker that outlived its lease writes after a rival finished: no effect."""
    (doc,) = await _documents(sessions, moved_to)
    before = tuple(await _row(sessions, doc))

    async with sessions.begin() as s:
        assert not await retry_later(s, EMBEDDING, doc, timedelta(minutes=10))
        assert not await release(s, EMBEDDING, doc)
        assert not await fail(s, EMBEDDING, doc, FailureReason.EMBEDDING_FAILED)

    assert tuple(await _row(sessions, doc)) == before


async def test_the_hand_off_starts_embedding_at_once_with_a_fresh_budget(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    (doc,) = await _documents(sessions, DocumentStatus.PENDING)
    await _claim(sessions)
    await _expire(sessions, doc)
    await _claim(sessions)  # extraction's second attempt; its lease still runs

    async with sessions.begin() as s:
        await s.execute(
            guarded_update(doc, FINISH_EXTRACTION).values(raw_text="text", **hand_off())
        )

    claimed = await _claim(sessions, EMBEDDING)
    assert claimed is not None and (claimed.document_id, claimed.attempt) == (doc, 1)


@pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5, 10])
def test_backoff_is_full_jitter_under_an_exponential_cap(attempt: int) -> None:
    base, cap = timedelta(seconds=30), timedelta(seconds=600)
    ceiling = min(cap, base * 2 ** (attempt - 1))

    assert backoff(attempt, base, cap, rng=lambda: 0.0) == timedelta()
    assert backoff(attempt, base, cap, rng=lambda: 0.999999) <= ceiling
    assert backoff(attempt, base, cap, rng=lambda: 0.5) == ceiling / 2


async def test_the_claim_reads_only_owed_rows_through_their_index(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    sql = claim_statement(EXTRACTION, lease=LEASE, max_attempts=BUDGET).compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    )
    async with sessions.begin() as s:
        # On a near-empty table the planner would scan it whole; forbid that to see
        # whether the partial index fits the claim's WHERE at all.
        await s.execute(sa.text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join((await s.execute(sa.text(f"EXPLAIN {sql}"))).scalars())

    assert "ix_documents_owed_available_at" in plan
