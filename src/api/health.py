"""``/health``: whether this API process is ready to serve (ADR-028).

The status code comes from what the API itself needs, the database and the cache.
The pipeline's lag is reported beside it and never fails it: a backlog is the
workers' to clear, and taking the API out of rotation would not clear it.
"""

import asyncio
from typing import Any

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.workers.queue import EMBEDDING, EXTRACTION, lag_statement

router = APIRouter(tags=["ops"])

# Each dependency gets this long: a probe that hangs tells its caller less than one
# that fails (asyncpg alone waits 60 s for a server that never answers).
PROBE_TIMEOUT_S = 3.0

# At connect, the driver's own errors arrive unwrapped by SQLAlchemy: OSError when
# no connection can be made, asyncpg's when the server refuses it (a wrong password,
# a missing database, the connection limit).
_DATABASE_ERRORS = (
    SQLAlchemyError,
    OSError,
    asyncpg.PostgresError,
    asyncpg.InterfaceError,
    TimeoutError,
)


async def _database(engine: Any) -> tuple[str, dict[str, float] | str]:
    """The database check and, on the same connection, how long due work has
    waited for a worker per phase: the queue's own notion of "due", so it counts
    exactly what a claim would take."""
    database = "error"
    pipeline: dict[str, float] | str = "error"
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_S), engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            database = "ok"
            pipeline = {
                f"{phase.name}_lag_s": round(await conn.scalar(lag_statement(phase)), 1)
                for phase in (EXTRACTION, EMBEDDING)
            }
    except _DATABASE_ERRORS:
        # Either check left as "error"; a database that failed isn't asked again.
        pass
    return database, pipeline


async def _cache() -> str:
    """The cache and rate limiter's Redis (REDIS_URL), through the app's own client."""
    from src.core.redis import get_async_redis_client

    try:
        client = get_async_redis_client()
    except RuntimeError:
        return "unavailable"
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_S):
            await client.ping()
    except RedisError, TimeoutError:
        return "error"
    return "ok"


@router.get("/health", response_model=None)
async def health(request: Request) -> JSONResponse:
    checks: dict[str, str] = {}
    pipeline: dict[str, float] | str

    engine: Any = getattr(request.app.state, "engine", None)
    if engine is None:
        checks["database"], pipeline = "unavailable", "unavailable"
    else:
        checks["database"], pipeline = await _database(engine)
    checks["cache"] = await _cache()

    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "unhealthy",
            "checks": checks,
            "pipeline": pipeline,
        },
    )
