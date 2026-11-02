"""The acceptance suite's world (P7): a scratch database, a fake embedding provider,
supervised worker processes and the in-process API. The pieces live in
tests/support/acceptance.py."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.api.v1 import documents as documents_api
from src.core.config import settings
from src.core.dependencies import get_db, get_http_client
from src.core.redis import get_redis
from src.main import app
from tests.support.acceptance import (
    AcceptanceDatabase,
    Api,
    Observer,
    Pipeline,
    Supervisor,
)
from tests.support.database import migrate, scratch_database
from tests.support.fake_gemini import FakeGemini


@pytest.fixture(scope="module")
def acceptance_db(
    postgres_container: PostgresContainer,
) -> Iterator[AcceptanceDatabase]:
    """Its own database: a claim takes the oldest owed row of the whole table."""
    with scratch_database(postgres_container, "acceptance") as url:
        migrate(url)
        yield AcceptanceDatabase(
            name=sa.make_url(url).database or "",
            url=url.replace("postgresql+psycopg", "postgresql+asyncpg"),
            admin=url,
            cluster=sa.make_url(postgres_container.get_connection_url())
            .set(drivername="postgresql+psycopg")
            .render_as_string(hide_password=False),
        )


@pytest.fixture
async def pipeline(
    acceptance_db: AcceptanceDatabase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Pipeline]:
    admin = sa.create_engine(acceptance_db.admin)
    with admin.begin() as conn:
        conn.execute(sa.text("TRUNCATE documents, workspaces, users CASCADE"))
    admin.dispose()

    uploads, worker_cwd, logs = (tmp_path / d for d in ("uploads", "cwd", "logs"))
    for d in (uploads, worker_cwd, logs):
        d.mkdir()
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(uploads))

    provider = FakeGemini(settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS)
    workers = Supervisor(acceptance_db.url, provider.url, worker_cwd, logs)
    observer = Observer(acceptance_db.url)

    engine = create_async_engine(acceptance_db.url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def committing_db() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected outbound HTTP call: {request.url}")

    fake_redis = FakeRedis()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
    app.dependency_overrides[get_db] = committing_db
    app.dependency_overrides[get_redis] = lambda: fake_redis
    app.dependency_overrides[get_http_client] = lambda: http_client
    # Scenarios upload more than the per-minute cap; rate limiting is tested elsewhere.
    app.dependency_overrides[documents_api._upload_rate_limit] = lambda: None
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    try:
        api = Api(client)
        await api.sign_in()
        yield Pipeline(api, workers, provider, observer, acceptance_db)
    finally:
        workers.close()
        provider.close()
        await client.aclose()
        app.dependency_overrides.clear()
        await http_client.aclose()
        await fake_redis.aclose()
        await observer.dispose()
        await engine.dispose()
