"""/health on a real PostgreSQL (ADR-028): the status code is the API's readiness
(database and cache), the pipeline's lag is reported beside it and never fails it.

A scratch database of its own: lag is a value of the whole documents table.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Generator
from datetime import timedelta
from typing import Any

import pytest
import redis.asyncio as aioredis
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import NullPool, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

import src.core.redis
from src.api import health
from src.domain.documents import DocumentStatus
from src.main import app
from tests.integration.test_work_queue import _documents
from tests.support.database import migrate, scratch_database


@pytest.fixture(scope="module")
def health_db(postgres_container: PostgresContainer) -> Generator[str]:
    with scratch_database(postgres_container, "health") as url:
        migrate(url)
        yield url.replace("postgresql+psycopg", "postgresql+asyncpg")


@pytest.fixture
async def engine(health_db: str) -> AsyncGenerator[AsyncEngine]:
    engine = create_async_engine(health_db, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(sa.text("TRUNCATE documents, workspaces, users CASCADE"))
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """The app's Redis client, as the lifespan would set it."""
    fake = FakeRedis()
    monkeypatch.setattr(src.core.redis, "_async_client", fake)
    return fake


async def _health(monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine) -> Any:
    monkeypatch.setattr(app.state, "engine", engine, raising=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        response = await c.get("/health")
    return response.status_code, response.json()


async def test_a_ready_api_answers_200_and_reports_each_phases_lag(
    monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine, cache: FakeRedis
) -> None:
    await _documents(
        async_sessionmaker(engine),
        DocumentStatus.PENDING,
        DocumentStatus.READY,
        waited=[timedelta(hours=1), timedelta(hours=2)],
    )

    status, body = await _health(monkeypatch, engine)

    assert status == 200
    assert body["status"] == "healthy"
    assert body["checks"] == {"database": "ok", "cache": "ok"}
    assert 3600 <= body["pipeline"]["extract_lag_s"] < 3660
    assert 7200 <= body["pipeline"]["embed_lag_s"] < 7260


async def test_lag_is_information_a_lagging_pipeline_stays_200(
    monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine, cache: FakeRedis
) -> None:
    """A week-old upload no worker took: the API itself is still ready."""
    await _documents(
        async_sessionmaker(engine), DocumentStatus.PENDING, waited=[timedelta(days=7)]
    )

    status, body = await _health(monkeypatch, engine)

    assert (status, body["status"]) == (200, "healthy")
    assert body["pipeline"]["extract_lag_s"] >= 7 * 24 * 3600


async def test_a_failed_lag_query_is_reported_and_leaves_the_status_alone(
    monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine, cache: FakeRedis
) -> None:
    # A real database error, raised by PostgreSQL itself.
    monkeypatch.setattr(health, "lag_statement", lambda _: sa.text("SELECT 1/0"))

    status, body = await _health(monkeypatch, engine)

    assert (status, body["status"]) == (200, "healthy")
    assert body["pipeline"] == "error"


@pytest.fixture
async def refusing(health_db: str) -> AsyncGenerator[dict[str, str]]:
    """Database URLs that PostgreSQL (or the OS) refuses, each a different way: the
    driver raises its own errors at connect, which SQLAlchemy doesn't wrap."""
    url = make_url(health_db)
    admin = create_async_engine(health_db, poolclass=NullPool)
    async with admin.begin() as conn:
        await conn.execute(sa.text("DROP ROLE IF EXISTS health_limited"))
        await conn.execute(
            sa.text("CREATE ROLE health_limited LOGIN PASSWORD 'x' CONNECTION LIMIT 0")
        )
    try:
        yield {
            "port-closed": url.set(host="127.0.0.1", port=1).render_as_string(False),
            "wrong-password": url.set(password="wrong").render_as_string(False),
            "missing-database": url.set(database="no_such_db").render_as_string(False),
            "connection-limit": url.set(
                username="health_limited", password="x"
            ).render_as_string(False),
        }
    finally:
        async with admin.begin() as conn:
            await conn.execute(sa.text("DROP ROLE health_limited"))
        await admin.dispose()


@pytest.mark.parametrize(
    "how", ["port-closed", "wrong-password", "missing-database", "connection-limit"]
)
async def test_a_database_that_refuses_the_api_is_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    cache: FakeRedis,
    refusing: dict[str, str],
    how: str,
) -> None:
    refused = create_async_engine(refusing[how], poolclass=NullPool)
    try:
        status, body = await _health(monkeypatch, refused)
    finally:
        await refused.dispose()

    assert (status, body["status"]) == (503, "unhealthy")
    assert body["checks"] == {"database": "error", "cache": "ok"}
    assert body["pipeline"] == "error"


@pytest.fixture
async def silent_server() -> AsyncGenerator[int]:
    """A port that accepts connections and never answers: a hung server, or a
    network that drops packets, without depending on one."""
    accepted: list[asyncio.StreamWriter] = []

    async def hold(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.append(writer)

    server = await asyncio.start_server(hold, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        for writer in accepted:
            writer.close()
        server.close()


async def test_a_database_that_never_answers_fails_within_one_bound(
    monkeypatch: pytest.MonkeyPatch, cache: FakeRedis, silent_server: int
) -> None:
    """One bounded wait for the database, not one per query: once it has failed,
    the lag query isn't tried."""
    monkeypatch.setattr(health, "PROBE_TIMEOUT_S", 1.0)
    silent = create_async_engine(
        f"postgresql+asyncpg://u:p@127.0.0.1:{silent_server}/x", poolclass=NullPool
    )
    started = time.monotonic()
    try:
        status, body = await asyncio.wait_for(_health(monkeypatch, silent), 10)
    finally:
        await silent.dispose()

    assert time.monotonic() - started < 1.8
    assert (status, body["checks"]["database"], body["pipeline"]) == (
        503,
        "error",
        "error",
    )


async def test_a_cache_that_never_answers_fails_within_the_bound(
    monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine, silent_server: int
) -> None:
    monkeypatch.setattr(health, "PROBE_TIMEOUT_S", 1.0)
    silent = aioredis.Redis(host="127.0.0.1", port=silent_server)
    monkeypatch.setattr(src.core.redis, "_async_client", silent)
    started = time.monotonic()
    try:
        status, body = await asyncio.wait_for(_health(monkeypatch, engine), 10)
    finally:
        await silent.aclose()

    assert time.monotonic() - started < 1.8
    assert (status, body["checks"]) == (503, {"database": "ok", "cache": "error"})


async def test_before_startup_nothing_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No engine and no cache client yet (the lifespan hasn't run)."""
    monkeypatch.setattr(src.core.redis, "_async_client", None)
    monkeypatch.delattr(app.state, "engine", raising=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        response = await c.get("/health")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "unavailable",
        "cache": "unavailable",
    }
    assert response.json()["pipeline"] == "unavailable"


async def test_an_unreachable_cache_is_unhealthy(
    monkeypatch: pytest.MonkeyPatch, engine: AsyncEngine
) -> None:
    monkeypatch.setattr(src.core.redis, "_async_client", FakeRedis(connected=False))

    status, body = await _health(monkeypatch, engine)

    assert (status, body["checks"]["cache"]) == (503, "error")
