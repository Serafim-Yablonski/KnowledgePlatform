"""Global test configuration and shared fixtures."""

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import NullPool, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.composition import Services, build_services
from src.core.observability import configure_logfire
from src.main import app
from tests.support.connections import EmbeddingProvider, HeldConnections, Outbound
from tests.support.docker import wait_for_port_mappings
from tests.support.lifespan import ServedWithState


@pytest.fixture(scope="session", autouse=True)
def _api_tracing_config() -> None:
    """Run every test under the API process's tracing config, not Logfire's
    unconfigured default, which trusts any incoming trace context. Never exports:
    no token, whatever .env says."""
    configure_logfire(
        token=None,
        environment="test",
        service_name="knowledge-platform",
        trust_incoming_trace_context=False,
    )


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
    container = PostgresContainer("pgvector/pgvector:pg18")
    # Covers the Ryuk reaper's port lookup too: start() launches the reaper first.
    with wait_for_port_mappings():
        container.start()
    try:
        yield container
    finally:
        container.stop()


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
    """Function-scoped session backed by a SAVEPOINT, for repository tests.

    Each test runs inside a nested transaction; the outer transaction is rolled
    back after the test, resetting DB state without truncating tables. Faithful for
    a repository, which only ever runs inside its caller's transaction (ADR-033);
    service, REST and MCP tests commit for real (``committing_engine``).
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
def settings_override(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Patch individual Settings fields for a single test.

    Usage: settings_override.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    ``settings`` is the instance ``get_settings()`` caches, so a patch reaches both.
    Never clear that cache here: the next ``get_settings()`` would build a new
    instance, and patches to ``settings`` would no longer reach its callers.
    """
    return monkeypatch


# ---------------------------------------------------------------------------
# Real commits (ADR-033; Track 2.5 D6 (b)). Service, REST and MCP tests run the
# production wiring with real commits; repository tests keep ``db_session``.
# ---------------------------------------------------------------------------

_TABLES = (
    "users, workspaces, workspace_memberships, documents, document_chunks, api_keys"
)


async def _truncate(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        # A test that leaked an open transaction would make TRUNCATE wait for it
        # forever, hanging the suite; this fails it instead.
        await conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
        await conn.execute(sa.text(f"TRUNCATE {_TABLES} CASCADE"))


@pytest.fixture
async def committing_engine(test_engine: AsyncEngine) -> AsyncGenerator[AsyncEngine]:
    """The test database with real commits, emptied before and after the test."""
    await _truncate(test_engine)
    try:
        yield test_engine
    finally:
        await _truncate(test_engine)


@pytest.fixture
def held_connections(test_engine: AsyncEngine) -> Generator[HeldConnections]:
    """Connections the test engine has checked out at any moment."""
    counter = HeldConnections()
    pool = test_engine.sync_engine.pool
    event.listen(pool, "checkout", counter.checkout)
    event.listen(pool, "checkin", counter.checkin)
    try:
        yield counter
    finally:
        event.remove(pool, "checkout", counter.checkout)
        event.remove(pool, "checkin", counter.checkin)


@pytest.fixture
def embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()


@pytest.fixture
async def provider_http(
    embedding_provider: EmbeddingProvider,
) -> AsyncGenerator[httpx.AsyncClient]:
    """The app's shared HTTP client, its only outbound peer the fake provider."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(embedding_provider.handle)
    ) as client:
        yield client


@pytest.fixture
async def fake_redis() -> AsyncGenerator[FakeRedis]:
    redis = FakeRedis()
    try:
        yield redis
    finally:
        await redis.aclose()


@pytest.fixture
async def committing_services(
    committing_engine: AsyncEngine,
    fake_redis: FakeRedis,
    provider_http: httpx.AsyncClient,
) -> Services:
    """The services as the composition root builds them, on the committing database:
    each method opens its own unit of work (ADR-033)."""
    return build_services(committing_engine, fake_redis, provider_http)


@asynccontextmanager
async def _rest_client(
    services: Services, redis: FakeRedis, engine: AsyncEngine
) -> AsyncIterator[AsyncClient]:
    # What the lifespan would hand every request (src.main.LifespanState).
    served = ServedWithState(
        app, {"services": services, "redis": redis, "engine": engine}
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=served), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
async def committing_client(
    committing_services: Services,
    fake_redis: FakeRedis,
    committing_engine: AsyncEngine,
) -> AsyncGenerator[AsyncClient]:
    """The REST API on the committing database; its only outbound peer is the fake
    embedding provider."""
    async with _rest_client(committing_services, fake_redis, committing_engine) as c:
        yield c


@pytest.fixture
def outbound() -> Outbound:
    return Outbound()


@pytest.fixture
async def async_client(
    committing_engine: AsyncEngine, fake_redis: FakeRedis, outbound: Outbound
) -> AsyncGenerator[AsyncClient]:
    """The REST API on the committing database; an outbound call fails the test
    unless it sets ``outbound.handler``."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(outbound.handle)
    ) as http:
        services = build_services(committing_engine, fake_redis, http)
        async with _rest_client(services, fake_redis, committing_engine) as client:
            yield client


@pytest.fixture
async def committing_session(
    committing_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession]:
    """Direct access to the committing database, for seeding rows and changing
    them behind the services' back; writes are seen by the app once committed."""
    async with AsyncSession(committing_engine, expire_on_commit=False) as session:
        yield session
