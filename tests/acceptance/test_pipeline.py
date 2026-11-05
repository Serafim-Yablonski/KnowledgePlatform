"""R-a, accepted (P7c): every upload reaches INDEXED or FAILED without operator
action, whatever happens to the workers, the database or the embedding provider.

Each scenario injects one failure from outside — a kill, a database outage, a
provider that errs or hangs — and waits for the upload to settle in real time.
``Pipeline.settle`` then checks the end state every scenario shares (one chunk set,
a reason for every failure, no worker dead of a failure it should survive).
Failures one mechanism handles alone (a parse timeout, a crash loop, racing claims,
a stop) are tested at that mechanism: tests/integration/test_jobs.py,
test_worker_process.py and test_work_queue.py. Harness: tests/support/acceptance.py.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from functools import cache

import pytest

from tests.pdfs import pdf_with_text
from tests.support.acceptance import (
    FAKE_API_KEY,
    PIPELINE_ENV,
    DocumentRow,
    Pipeline,
    eventually,
    parse_children,
)
from tests.support.fake_gemini import Reply

MAX_ATTEMPTS = int(PIPELINE_ENV["PIPELINE_MAX_ATTEMPTS"])
EMBED_SLOTS = int(PIPELINE_ENV["PIPELINE_EMBED_CONCURRENCY"])
EMBED_TIMEOUT = float(PIPELINE_ENV["PIPELINE_EMBED_TIMEOUT"])
# The worker's HTTP client gives up on a silent response after this (src/workers/
# __main__.py); a provider that trickles bytes never trips it.
CLIENT_READ_TIMEOUT = 30.0


@cache
def slow_pdf() -> bytes:
    """Parses in ~2.4 s (9000 short pages; measured in P7c): long enough to strike
    mid-parse, inside the harness's parse timeout, and little text to embed."""
    return pdf_with_text(*[f"p{i}" for i in range(9000)])


async def _upload_slow_pdf(pipeline: Pipeline) -> uuid.UUID:
    return await pipeline.api.upload("slow.pdf", slow_pdf(), "application/pdf")


async def _upload_text(pipeline: Pipeline, name: str = "notes.txt") -> uuid.UUID:
    return await pipeline.api.upload(name, f"The text of {name}.".encode())


async def _parsing(pipeline: Pipeline) -> None:
    async def running() -> bool | None:
        return True if parse_children() else None

    await eventually(running, timeout=30, what=pipeline.state("a PDF parse to start"))


async def _provider_requests(pipeline: Pipeline, count: int) -> None:
    async def received() -> bool | None:
        return True if len(pipeline.provider.requests) >= count else None

    await eventually(
        received, timeout=30, what=pipeline.state(f"{count} provider requests")
    )


async def _row(pipeline: Pipeline, doc: uuid.UUID) -> tuple[str, int, bool]:
    row: DocumentRow = await pipeline.observer.row(doc)
    return row.status, row.attempts, row.claimable


async def test_s1_uploads_are_indexed_through_the_real_workers(
    pipeline: Pipeline,
) -> None:
    docs = [
        await _upload_text(pipeline),
        await pipeline.api.upload(
            "page.pdf", pdf_with_text("A page of text."), "application/pdf"
        ),
    ]
    pipeline.workers.start("extract")
    pipeline.workers.start("embed")

    rows = await pipeline.settle(docs)

    assert [rows[d].status for d in docs] == ["INDEXED", "INDEXED"]
    assert [r.api_key for r in pipeline.provider.requests] == [FAKE_API_KEY] * 2


async def test_s2_a_worker_killed_mid_extraction_is_replaced_and_the_upload_indexed(
    pipeline: Pipeline,
) -> None:
    doc = await _upload_slow_pdf(pipeline)
    [extract] = pipeline.workers.start("extract")
    pipeline.workers.start("embed")
    await _parsing(pipeline)

    pipeline.workers.kill(extract)

    # Leased, the attempt counted: nobody writes an outcome, the lease will run out.
    assert await _row(pipeline, doc) == ("PROCESSING", 1, False)

    rows = await pipeline.settle([doc])
    assert rows[doc].status == "INDEXED"


async def test_s3_a_worker_killed_mid_embedding_is_replaced_and_the_upload_indexed(
    pipeline: Pipeline,
) -> None:
    working = threading.Event()
    pipeline.provider.script(Reply(hold=working))
    doc = await _upload_text(pipeline)
    pipeline.workers.start("extract")
    [embed] = pipeline.workers.start("embed")
    await _provider_requests(pipeline, 1)

    pipeline.workers.kill(embed)
    working.set()

    assert await _row(pipeline, doc) == ("READY", 1, False)
    rows = await pipeline.settle([doc])
    assert rows[doc].status == "INDEXED"
    assert len(pipeline.provider.requests) == 2


@pytest.mark.parametrize("phase", ["extract", "embed"])
async def test_s4_a_database_outage_mid_attempt_costs_an_attempt_not_the_upload(
    pipeline: Pipeline, phase: str
) -> None:
    working, retrying = threading.Event(), threading.Event()
    if phase == "extract":
        doc = await _upload_slow_pdf(pipeline)
    else:
        pipeline.provider.script(Reply(hold=working), Reply(hold=retrying))
        doc = await _upload_text(pipeline)
    pipeline.workers.start("extract")
    pipeline.workers.start("embed")
    if phase == "extract":
        await _parsing(pipeline)
    else:
        await _provider_requests(pipeline, 1)

    with pipeline.database_outage():
        working.set()

        # The attempt ends while the database is gone: its writes fail, and the
        # workers keep polling instead of dying.
        async def unrecorded() -> bool | None:
            log = pipeline.workers.log_text()
            return True if "could not record the outcome" in log else None

        await eventually(
            unrecorded, timeout=30, what=pipeline.state("the outcome write to fail")
        )
        await asyncio.sleep(1)

    # The lease returns the document, and the next claim counts the attempt the
    # outage cut off. Extraction's count is reset when it finishes, so it is read
    # mid-attempt: the slow PDF parses for seconds, embedding waits on the fake.
    owed = "PROCESSING" if phase == "extract" else "READY"

    async def reclaimed() -> bool | None:
        return True if await _row(pipeline, doc) == (owed, 2, False) else None

    await eventually(
        reclaimed, timeout=30, what=pipeline.state("the second attempt's claim")
    )
    retrying.set()

    rows = await pipeline.settle([doc])
    assert rows[doc].status == "INDEXED"
    assert "claim failed; retrying" in pipeline.workers.log_text()


async def test_s7_a_provider_that_errs_then_recovers_indexes_the_upload(
    pipeline: Pipeline,
) -> None:
    # The client retries a 5xx twice inside one call: three fail an attempt.
    pipeline.provider.script(*[Reply(status=503)] * 3)
    doc = await _upload_text(pipeline)
    pipeline.workers.start("extract")
    pipeline.workers.start("embed")

    rows = await pipeline.settle([doc])

    assert (rows[doc].status, rows[doc].attempts) == ("INDEXED", 2)
    assert len(pipeline.provider.requests) == 4


async def test_s7_a_provider_that_keeps_erring_fails_the_upload(
    pipeline: Pipeline,
) -> None:
    pipeline.provider.script(*[Reply(status=503)] * (3 * MAX_ATTEMPTS))
    doc = await _upload_text(pipeline)
    pipeline.workers.start("extract")
    pipeline.workers.start("embed")

    rows = await pipeline.settle([doc])

    assert (rows[doc].status, rows[doc].failure_reason, rows[doc].attempts) == (
        "FAILED",
        "EMBEDDING_FAILED",
        MAX_ATTEMPTS,
    )
    assert len(pipeline.provider.requests) == 3 * MAX_ATTEMPTS


async def test_s8_a_provider_that_hangs_costs_attempts_not_the_workers_slots(
    pipeline: Pipeline,
) -> None:
    """A hang on every slot of the worker. Each attempt's timeout ends its own wait
    and frees its slot; without it the lease would hand the document to the next
    slot, until all of them hung, and only the client's read timeout would free one
    (never, for a provider that trickles bytes)."""
    never = threading.Event()
    pipeline.provider.script(*[Reply(hold=never)] * EMBED_SLOTS)
    doc = await _upload_text(pipeline)
    pipeline.workers.start("extract")
    pipeline.workers.start("embed")

    # Each hang lasts one attempt timeout; the margin covers the extraction,
    # backoffs and polls, and stays well below the client's read timeout.
    bound = EMBED_SLOTS * EMBED_TIMEOUT + 10
    assert bound < CLIENT_READ_TIMEOUT
    rows = await pipeline.settle([doc], timeout=bound)

    assert (rows[doc].status, rows[doc].attempts) == ("INDEXED", EMBED_SLOTS + 1)
    assert len(pipeline.provider.requests) == EMBED_SLOTS + 1
