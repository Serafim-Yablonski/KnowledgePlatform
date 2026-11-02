import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(tags=["ops"])


def _connect_to_broker() -> None:
    """Open and close one broker connection; raise if it can't be made."""
    from src.workers.celery_app import celery_app

    with celery_app.connection_for_write() as conn:
        conn.ensure_connection(max_retries=0)  # one attempt: a probe, not a client


@router.get("/health", response_model=None)
async def health(request: Request) -> JSONResponse:
    checks: dict[str, str] = {}

    engine: Any = getattr(request.app.state, "engine", None)
    if engine is None:
        checks["database"] = "unavailable"
    else:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except SQLAlchemyError:
            checks["database"] = "error"

    # The cache and rate limiter's Redis (REDIS_URL), through the app's own client.
    try:
        from src.core.redis import get_async_redis_client

        await get_async_redis_client().ping()
        checks["cache"] = "ok"
    except RedisError:
        checks["cache"] = "error"
    except RuntimeError:
        checks["cache"] = "unavailable"

    # The broker (CELERY_BROKER_URL) can be another server than the cache: uploads
    # need it. kombu's transport is synchronous, hence the executor (as below).
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _connect_to_broker)
        checks["broker"] = "ok"
    except Exception:  # noqa: BLE001 — a health probe reports failure, never raises
        checks["broker"] = "error"

    # Ping Celery workers via the broker. run_in_executor avoids blocking the
    # async handler — inspect() uses a synchronous kombu transport internally.
    try:
        from src.workers.celery_app import celery_app

        ping_result: Any = await loop.run_in_executor(
            None,
            lambda: celery_app.control.inspect(timeout=1.0).ping(),
        )
        checks["workers"] = "ok" if ping_result else "unavailable"
    except Exception:  # noqa: BLE001 — a health probe reports failure, never raises
        checks["workers"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "healthy" if all_ok else "unhealthy", "checks": checks},
    )
