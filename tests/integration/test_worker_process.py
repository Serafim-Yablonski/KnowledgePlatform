"""Pipeline workers as real processes (ADR-027): what a stop, a kill and a crash loop
leave in the documents table.

A scratch database of its own (a claim takes the oldest owed row of the whole table).
Most tests run tests/support/pipeline_worker.py: the real loop, signals and writes,
with a stand-in job. Time moves by writing ``available_at``, never by waiting out a
lease.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen

import pytest
import sqlalchemy as sa
from testcontainers.postgres import PostgresContainer

from src.core.config import settings
from tests.support.database import migrate, scratch_database

ROOT = Path(__file__).resolve().parents[2]
GRACE_S = 2.0


@pytest.fixture(scope="module")
def worker_db(postgres_container: PostgresContainer) -> Generator[str]:
    with scratch_database(postgres_container, "workers") as url:
        migrate(url)
        yield url


@pytest.fixture
def db(worker_db: str) -> Generator[sa.Engine]:
    engine = sa.create_engine(worker_db)
    with engine.begin() as conn:
        conn.execute(sa.text("TRUNCATE documents, workspaces, users CASCADE"))
    try:
        yield engine
    finally:
        engine.dispose()


def _document(db: sa.Engine, file_path: str = "/nonexistent") -> uuid.UUID:
    user, ws, doc = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with db.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, :email, 'x')"
            ),
            {"id": user, "email": f"{user}@example.com"},
        )
        conn.execute(
            sa.text(
                "INSERT INTO workspaces (id, name, slug, created_by) "
                "VALUES (:id, 'W', :slug, :user)"
            ),
            {"id": ws, "slug": f"w-{ws.hex[:8]}", "user": user},
        )
        conn.execute(
            sa.text(
                "INSERT INTO documents (id, workspace_id, title, content_type, "
                "file_path, file_size_bytes, uploaded_by) "
                "VALUES (:id, :ws, 'T', 'PLAINTEXT', :path, 1, :user)"
            ),
            {"id": doc, "ws": ws, "path": file_path, "user": user},
        )
    return doc


def _row(db: sa.Engine, doc: uuid.UUID) -> sa.Row:  # type: ignore[type-arg]
    with db.connect() as conn:
        return conn.execute(
            sa.text(
                "SELECT status, failure_reason, attempts, "
                "available_at <= now() AS claimable FROM documents WHERE id = :id"
            ),
            {"id": doc},
        ).one()


def _expire_lease(db: sa.Engine, doc: uuid.UUID) -> None:
    with db.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE documents SET available_at = now() - interval '1 second' "
                "WHERE id = :id"
            ),
            {"id": doc},
        )


def _env(worker_db: str, runs: Path) -> dict[str, str]:
    return {
        **os.environ,
        "LOGFIRE_TOKEN": "",  # never export from a test
        "DATABASE_URL": worker_db.replace("postgresql+psycopg", "postgresql+asyncpg"),
        "ASYNC_DATABASE_URL": "",
        "PIPELINE_POLL_INTERVAL": "0.05",
        "PIPELINE_SHUTDOWN_GRACE": str(GRACE_S),
        "KP_WORKER_TEST_DIR": str(runs),
    }


@contextmanager
def _worker(args: list[str], worker_db: str, runs: Path) -> Iterator[Popen[bytes]]:
    with (runs / "worker.log").open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", *args],
            cwd=ROOT,
            env=_env(worker_db, runs),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield proc
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def _support(job: str) -> list[str]:
    return ["tests.support.pipeline_worker", job]


def _runs(runs: Path) -> int:
    return len(list(runs.glob("run-*")))


def _wait(check: object, timeout: float = 30, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not check():  # type: ignore[operator]
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def _log(runs: Path) -> str:
    return (runs / "worker.log").read_text()


def test_sigterm_releases_a_running_attempt_and_exits_within_the_grace(
    db: sa.Engine, worker_db: str, tmp_path: Path
) -> None:
    doc = _document(db)
    with _worker(_support("block"), worker_db, tmp_path) as proc:
        _wait(lambda: _runs(tmp_path) == 1, what="the job to start")

        stopped_at = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=GRACE_S + 10)
        took = time.monotonic() - stopped_at

    assert code == 0, _log(tmp_path)
    assert GRACE_S <= took < GRACE_S + 5
    # Claimable at once, and the attempt the stop cut off doesn't count.
    assert tuple(_row(db, doc)) == ("PROCESSING", None, 0, True)


def test_a_killed_workers_document_is_reclaimed_once_its_lease_runs_out(
    db: sa.Engine, worker_db: str, tmp_path: Path
) -> None:
    doc = _document(db)
    with _worker(_support("block"), worker_db, tmp_path) as proc:
        _wait(lambda: _runs(tmp_path) == 1, what="the job to start")
        proc.kill()
        proc.wait()

    # No outcome written: leased, the attempt counted.
    assert tuple(_row(db, doc)) == ("PROCESSING", None, 1, False)

    _expire_lease(db, doc)
    with _worker(_support("finish"), worker_db, tmp_path) as proc:
        _wait(lambda: _runs(tmp_path) == 2, what="the reclaim")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=GRACE_S + 10) == 0

    assert _row(db, doc).attempts == 2


def test_a_crash_loop_ends_failed_as_interrupted(
    db: sa.Engine, worker_db: str, tmp_path: Path
) -> None:
    """Every attempt kills its worker: none can record an outcome. The claim that
    finds the budget used up fails the document instead of running it again."""
    doc = _document(db)
    for crash in range(1, settings.PIPELINE_MAX_ATTEMPTS + 1):
        with _worker(_support("crash"), worker_db, tmp_path) as proc:
            assert proc.wait(timeout=30) == -signal.SIGKILL, _log(tmp_path)
        assert _runs(tmp_path) == crash
        _expire_lease(db, doc)

    with _worker(_support("crash"), worker_db, tmp_path) as proc:
        _wait(lambda: _row(db, doc).status == "FAILED", what="FAILED")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=GRACE_S + 10) == 0

    assert tuple(_row(db, doc))[:3] == (
        "FAILED",
        "INTERRUPTED",
        settings.PIPELINE_MAX_ATTEMPTS,
    )
    assert _runs(tmp_path) == settings.PIPELINE_MAX_ATTEMPTS


def test_the_extract_worker_extracts_an_upload(
    db: sa.Engine, worker_db: str, tmp_path: Path
) -> None:
    """``python -m src.workers extract`` end to end: claim, the real job, its write."""
    path = tmp_path / "a.txt"
    path.write_text("hello")
    doc = _document(db, file_path=str(path))

    with _worker(["src.workers", "extract"], worker_db, tmp_path) as proc:
        _wait(lambda: _row(db, doc).status == "READY", what="READY")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=GRACE_S + 10) == 0, _log(tmp_path)


def test_an_idle_worker_traces_nothing(worker_db: str, tmp_path: Path) -> None:
    """Database calls are traced in the workers, but the claim polls every second:
    an empty poll must not start a trace."""
    script = textwrap.dedent(
        """
        import asyncio
        import sqlalchemy as sa
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from src.core.config import settings
        from src.core.observability import configure_logfire, instrument_libraries
        from src.workers.runner import Worker, extract_work

        configure_logfire(
            token=None, environment="test", service_name="kp",
            trust_incoming_trace_context=False,
        )
        instrument_libraries()
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

        async def main():
            engine = create_async_engine(settings.ASYNC_DATABASE_URL.get_secret_value())
            sessions = async_sessionmaker(engine)
            # The control: queries outside the claim are traced.
            async with sessions() as s:
                await s.execute(sa.text("SELECT 1"))
            print(f"control={len(exporter.get_finished_spans())}")
            exporter.clear()
            stop = asyncio.Event()
            asyncio.get_running_loop().call_later(0.5, stop.set)
            await Worker(sessions, extract_work(sessions, settings)).run(stop)
            await engine.dispose()

        asyncio.run(main())
        print(f"idle={len(exporter.get_finished_spans())}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=_env(worker_db, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    # Logfire's console prints spans to stdout too: pick out the counts.
    counts = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("control=", "idle="))
    )
    control, idle = int(counts["control"]), int(counts["idle"])
    # Ten polls at the test's interval, none traced.
    assert (control > 0, idle) == (True, 0), result.stdout
