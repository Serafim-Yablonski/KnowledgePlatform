"""Global test configuration and shared fixtures."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.core.config import get_settings
from src.core.dependencies import get_db
from src.core.redis import get_redis
from src.main import app


def _ensure_docker_host() -> None:
    """Point testcontainers at Docker Desktop's socket on macOS if needed."""
    if "DOCKER_HOST" not in os.environ:
        desktop_sock = Path.home() / ".docker" / "run" / "docker.sock"
        if desktop_sock.exists():
            os.environ["DOCKER_HOST"] = f"unix://{desktop_sock}"


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer]:
    _ensure_docker_host()
    # pgvector/pgvector:pg18 ships PostgreSQL 18 with pgvector pre-installed.
    with PostgresContainer("pgvector/pgvector:pg18") as container:
        yield container


@pytest.fixture(scope="session")
def test_db_url(postgres_container: PostgresContainer) -> str:
    url: str = postgres_container.get_connection_url()
    # testcontainers returns a psycopg2 URL; convert to asyncpg for SQLAlchemy async.
    return url.replace("psycopg2", "asyncpg")


@pytest.fixture(scope="session")
def apply_migrations(test_db_url: str) -> Generator[None]:
    from alembic.config import Config

    from alembic import command

    cfg = Config("alembic.ini")
    # Override URL so env.py's _get_url() uses the testcontainer database.
    cfg.set_main_option("sqlalchemy.url", test_db_url)
    # env.py calls asyncio.run() internally; safe here because no pytest-asyncio
    # event loop is running yet during session-scoped sync fixture setup.
    command.upgrade(cfg, "head")
    yield


@pytest.fixture(scope="session")
def test_engine(test_db_url: str, apply_migrations: None) -> Generator[AsyncEngine]:
    # NullPool ensures each test.connect() opens a fresh asyncpg connection in the
    # current event loop. Without it, a pooled connection from test N's loop gets
    # reused in test N+1's loop, causing "Future attached to a different loop".
    engine = create_async_engine(test_db_url, echo=False, poolclass=NullPool)
    yield engine
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(engine.dispose())
    finally:
        loop.close()


@pytest.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    """Function-scoped session backed by a SAVEPOINT.

    Each test runs inside a nested transaction; the outer transaction is rolled
    back after the test, resetting DB state without truncating tables.
    """
    async with test_engine.connect() as conn:
        await conn.begin()
        session = AsyncSession(
            bind=conn,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()


@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    # ASGITransport does not trigger the FastAPI lifespan, so app.state.redis is
    # never populated. Override get_redis with an in-memory fake so rate limiting
    # and caching work without a real Redis process.
    fake_redis = FakeRedis()

    async def _override_get_db() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_redis] = lambda: fake_redis
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        await fake_redis.aclose()


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Patch individual Settings fields for a single test.

    Usage: settings_override.setattr(settings, "SECRET_KEY", "test-value")
    """
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
