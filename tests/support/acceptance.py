"""The pipeline's acceptance harness (P7): R-a as a black box.

Uploads go through the API, workers run as ``python -m src.workers``, failures come
from outside. Three roles, kept apart:

- the world (``Supervisor``, ``FakeGemini`` from tests/support/fake_gemini.py):
  starts, kills and restarts worker processes, and scripts the embedding provider.
  A restart stands in for compose's ``restart: unless-stopped``
  (tests/unit/test_compose.py pins that policy);
- the API (fixtures in tests/acceptance/conftest.py): the in-process app on the
  scratch database, committing for real;
- the observer (``Observer``): reads the database, never writes it: the end
  state must come about without anyone's help.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import Popen
from typing import Any

import sqlalchemy as sa
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import PipelineSettings
from tests.support.fake_gemini import FakeGemini

ROOT = Path(__file__).resolve().parents[2]
PDF_CHILD = ROOT / "src" / "workers" / "pdf_text.py"
FAKE_API_KEY = "acceptance-fake-key"
TERMINAL = ("INDEXED", "FAILED")

# Short enough that leases run out within a test, long enough for a real attempt.
PIPELINE_ENV = {
    "PIPELINE_MAX_ATTEMPTS": "5",
    "PIPELINE_POLL_INTERVAL": "0.05",
    "PIPELINE_SHUTDOWN_GRACE": "1",
    "PIPELINE_EXTRACT_CONCURRENCY": "2",
    "PIPELINE_EMBED_CONCURRENCY": "2",
    # The slow PDF (tests/acceptance/test_pipeline.py) parses in ~2.4 s here: room
    # for a slower CI runner, and leases short enough to wait out.
    "PIPELINE_EXTRACT_PARSE_TIMEOUT": "6",
    "PIPELINE_EXTRACT_TIMEOUT": "7",
    "PIPELINE_EXTRACT_LEASE": "8",
    "PIPELINE_EMBED_TIMEOUT": "5",
    "PIPELINE_EMBED_LEASE": "6",
    "PIPELINE_EXTRACT_BACKOFF_BASE": "0.1",
    "PIPELINE_EMBED_BACKOFF_BASE": "0.1",
}
# Docker's first restart delay (it doubles from there; ours stays flat).
RESTART_DELAY_S = 0.1
# A worker process's start: interpreter, imports, engine.
STARTUP_S = 3.0


def worker_env(database_url: str, provider_url: str) -> dict[str, str]:
    """Everything a worker process sees: built from nothing, so neither the shell
    nor the developer's .env (the worker runs elsewhere) reaches it — no real key,
    no Logfire token, no PIPELINE_* override."""
    passed = {
        k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ
    }
    return {
        **passed,
        "PYTHONPATH": str(ROOT),
        "ENVIRONMENT": "test",
        "SECRET_KEY": "acceptance-secret-key-not-for-production",
        "DATABASE_URL": database_url,
        "EMBEDDING_BASE_URL": provider_url,
        "EMBEDDING_API_KEY": FAKE_API_KEY,
        **PIPELINE_ENV,
    }


def settle_deadline_s(env: Mapping[str, str] = PIPELINE_ENV) -> float:
    """How long an upload may take to settle under ``env``'s settings, at worst.

    Per phase, each of the attempts ends within its lease (recorded or not), then
    waits at most its backoff ceiling, a poll and, after a crash, a restart.
    Extraction's budget is reset when it finishes, so the phases add up.
    """
    cfg = PipelineSettings(
        _env_file=None,  # type: ignore[call-arg]
        **{k: v for k, v in env.items() if k.startswith("PIPELINE_")},
    )
    cap = 600.0  # runner.Work.backoff_cap

    def phase(lease: float, base: float) -> float:
        return sum(
            lease
            + min(cap, base * 2 ** (attempt - 1))
            + 1.5 * cfg.PIPELINE_POLL_INTERVAL
            + RESTART_DELAY_S
            + STARTUP_S
            for attempt in range(1, cfg.PIPELINE_MAX_ATTEMPTS + 1)
        )

    extract = phase(cfg.PIPELINE_EXTRACT_LEASE, cfg.PIPELINE_EXTRACT_BACKOFF_BASE)
    embed = phase(cfg.PIPELINE_EMBED_LEASE, cfg.PIPELINE_EMBED_BACKOFF_BASE)
    return extract + embed + STARTUP_S


async def eventually[T](
    check: Callable[[], Awaitable[T | None]],
    *,
    timeout: float,
    what: str | Callable[[], str],
    interval: float = 0.05,
) -> T:
    """Poll ``check`` until it returns something other than None. A callable ``what``
    is called only at the timeout, so it can describe the state then."""
    deadline = time.monotonic() + timeout
    while True:
        result = await check()
        if result is not None:
            return result
        assert time.monotonic() < deadline, (
            f"timed out after {timeout:.0f}s: {what() if callable(what) else what}"
        )
        await asyncio.sleep(interval)


# ─── Worker processes ─────────────────────────────────────────────────────────


@dataclass
class WorkerProcess:
    phase: str
    name: str
    log: Path
    proc: Popen[bytes] | None = None
    # The test killed it: its next exit is no crash.
    expected_exit: bool = False
    # Exited; restarted at this monotonic time.
    restart_at: float | None = None

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid


@dataclass
class Supervisor:
    """Runs worker processes and restarts any that exit, as compose's
    ``restart: unless-stopped`` does. An exit the test didn't cause is recorded
    in ``unexpected_exits``: a worker must never die of a failure it should
    survive."""

    database_url: str
    provider_url: str
    cwd: Path
    logs: Path
    workers: list[WorkerProcess] = field(default_factory=list)
    unexpected_exits: list[tuple[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def start(self, phase: str, count: int = 1) -> list[WorkerProcess]:
        started = []
        with self._lock:
            for _ in range(count):
                name = f"{phase}-{sum(w.phase == phase for w in self.workers) + 1}"
                worker = WorkerProcess(phase, name, self.logs / f"{name}.log")
                self._spawn(worker)
                self.workers.append(worker)
                started.append(worker)
        return started

    def kill(self, worker: WorkerProcess) -> None:
        """SIGKILL (the OOM killer, a lost host); restarted like any exit."""
        with self._lock:
            worker.expected_exit = True
            os.kill(worker.pid, signal.SIGKILL)

    def close(self) -> None:
        self._closing.set()
        self._thread.join()
        for worker in self.workers:
            if worker.proc is not None and worker.proc.poll() is None:
                worker.proc.kill()
                worker.proc.wait()

    def log_text(self) -> str:
        return "\n".join(
            f"--- {w.name} ---\n{w.log.read_text(errors='replace')}"
            for w in self.workers
            if w.log.exists()
        )

    def popen_kwargs(self) -> dict[str, Any]:
        """Where and with what a worker starts: built at each spawn, so nothing the
        test process's environment gains later reaches a worker either."""
        return {
            "cwd": self.cwd,
            "env": worker_env(self.database_url, self.provider_url),
        }

    def _spawn(self, worker: WorkerProcess) -> None:
        with worker.log.open("ab") as log:
            worker.proc = subprocess.Popen(
                [sys.executable, "-m", "src.workers", worker.phase],
                stdout=log,
                stderr=subprocess.STDOUT,
                **self.popen_kwargs(),
            )
        worker.expected_exit = False

    def _watch(self) -> None:
        # One pass under the lock decides and acts, so a stop can't slip in between
        # seeing an exit and restarting.
        while not self._closing.wait(0.05):
            with self._lock:
                now = time.monotonic()
                for worker in self.workers:
                    if worker.proc is None:
                        continue
                    if worker.restart_at is None:
                        if worker.proc.poll() is None:
                            continue
                        if not worker.expected_exit:
                            self.unexpected_exits.append(
                                (worker.name, worker.proc.returncode)
                            )
                        worker.restart_at = now + RESTART_DELAY_S
                    elif now >= worker.restart_at:
                        worker.restart_at = None
                        self._spawn(worker)


def processes() -> str:
    """Every running process, one ``pid args`` line each, never cut.

    ``-ww``: unlimited width. Linux's ps cuts ``args`` at ``COLUMNS`` even into a
    pipe, and GNU readline, which pytest imports early, sets COLUMNS=80 in the C
    environment (not ``os.environ``) that ps inherits: a parse child's path, past
    column 80 in a CI checkout, would never match.
    """
    return subprocess.run(
        ["ps", "-ww", "-A", "-o", "pid=", "-o", "args="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def parse_children() -> list[int]:
    """Every running PDF parse child."""
    return [
        int(line.split(None, 1)[0])
        for line in processes().splitlines()
        if str(PDF_CHILD) in line
    ]


# ─── The observer ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentRow:
    status: str
    failure_reason: str | None
    attempts: int
    claimable: bool


class Observer:
    """The database as the test sees it: read only."""

    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url, poolclass=NullPool)

    async def row(self, document_id: uuid.UUID) -> DocumentRow:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.text(
                        "SELECT status, failure_reason, attempts, "
                        "available_at <= now() AS claimable "
                        "FROM documents WHERE id = :id"
                    ),
                    {"id": document_id},
                )
            ).one()
        return DocumentRow(*row)

    async def chunks(self, document_id: uuid.UUID) -> tuple[list[int], int]:
        """The chunk indexes in order, and how many chunks lack an embedding."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.text(
                        "SELECT chunk_index, embedding IS NULL FROM document_chunks "
                        "WHERE document_id = :id ORDER BY chunk_index"
                    ),
                    {"id": document_id},
                )
            ).all()
        return [r[0] for r in rows], sum(r[1] for r in rows)

    async def dispose(self) -> None:
        await self.engine.dispose()


@dataclass(frozen=True)
class AcceptanceDatabase:
    name: str
    url: str  # asyncpg: the API, the workers and the observer
    admin: str  # psycopg: setup between tests
    cluster: str  # psycopg, the server's own database: outages


@contextmanager
def database_outage(db: AcceptanceDatabase) -> Iterator[None]:
    """The database goes away: every open connection is dropped and new ones are
    refused (a restart, a failover) until the block ends."""
    cluster = sa.create_engine(db.cluster, isolation_level="AUTOCOMMIT")
    try:
        with cluster.connect() as conn:
            conn.execute(sa.text(f'ALTER DATABASE "{db.name}" ALLOW_CONNECTIONS false'))
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name"
                ),
                {"name": db.name},
            )
        yield
    finally:
        with cluster.connect() as conn:
            conn.execute(sa.text(f'ALTER DATABASE "{db.name}" ALLOW_CONNECTIONS true'))
        cluster.dispose()


async def assert_settled(
    observer: Observer,
    documents: list[uuid.UUID],
    supervisor: Supervisor,
    max_attempts: int = int(PIPELINE_ENV["PIPELINE_MAX_ATTEMPTS"]),
) -> dict[uuid.UUID, DocumentRow]:
    """R-a's end state, checked after every scenario."""
    assert documents, "no documents to check"
    rows = {}
    for doc in documents:
        row = await observer.row(doc)
        rows[doc] = row
        assert row.status in TERMINAL, f"{doc} not settled: {row}"
        assert (row.status == "FAILED") == (row.failure_reason is not None), row
        assert 0 <= row.attempts <= max_attempts, row
        indexes, unembedded = await observer.chunks(doc)
        if row.status == "INDEXED":
            assert indexes == list(range(len(indexes))) and indexes, (doc, indexes)
            assert unembedded == 0, (doc, unembedded)
    assert supervisor.unexpected_exits == [], (
        f"workers died: {supervisor.unexpected_exits}\n{supervisor.log_text()}"
    )
    return rows


# ─── The API ──────────────────────────────────────────────────────────────────


@dataclass
class Api:
    """A client of the in-process app, signed in, with a workspace of its own."""

    client: Any  # httpx.AsyncClient
    workspace_id: str = ""

    async def sign_in(self) -> None:
        email = f"{uuid.uuid4().hex[:12]}@example.com"
        creds = {"email": email, "password": "acceptance-password"}
        resp = await self.client.post(
            "/api/v1/auth/register", json={**creds, "display_name": "Acceptance"}
        )
        assert resp.status_code == 201, resp.text
        resp = await self.client.post("/api/v1/auth/login", json=creds)
        assert resp.status_code == 200, resp.text
        self.client.headers["Authorization"] = f"Bearer {resp.json()['access_token']}"
        resp = await self.client.post("/api/v1/workspaces", json={"name": "Acceptance"})
        assert resp.status_code == 201, resp.text
        self.workspace_id = resp.json()["id"]

    async def upload(
        self, filename: str, content: bytes, content_type: str = "text/plain"
    ) -> uuid.UUID:
        resp = await self.client.post(
            f"/api/v1/workspaces/{self.workspace_id}/documents",
            files={"file": (filename, content, content_type)},
        )
        assert resp.status_code == 201, resp.text
        return uuid.UUID(resp.json()["id"])

    async def document(self, document_id: uuid.UUID) -> dict[str, Any]:
        """The document as a client sees it."""
        resp = await self.client.get(
            f"/api/v1/workspaces/{self.workspace_id}/documents/{document_id}"
        )
        assert resp.status_code == 200, resp.text
        body: dict[str, Any] = resp.json()
        return body


@dataclass
class Pipeline:
    """One scenario's world: the API, the workers, the provider and the observer."""

    api: Api
    workers: Supervisor
    provider: FakeGemini
    observer: Observer
    database: AcceptanceDatabase

    def database_outage(self) -> Any:
        return database_outage(self.database)

    def state(self, what: str) -> Callable[[], str]:
        """``what``, then the processes and the workers' logs at the time of the
        call: an ``eventually`` timeout's message."""
        return lambda: (
            f"{what}\n--- processes ---\n{processes()}{self.workers.log_text()}"
        )

    async def settle(
        self, documents: list[uuid.UUID], timeout: float | None = None
    ) -> dict[uuid.UUID, DocumentRow]:
        """Wait until the API shows every document settled, then check R-a's end
        state. The default bound is the settings' worst case."""

        async def all_settled() -> bool | None:
            for doc in documents:
                if (await self.api.document(doc))["status"].upper() not in TERMINAL:
                    return None
            return True

        await eventually(
            all_settled,
            timeout=timeout
            if timeout is not None
            else settle_deadline_s(self.workers.popen_kwargs()["env"]),
            what=self.state(f"{len(documents)} documents to settle"),
        )
        return await assert_settled(self.observer, documents, self.workers)
