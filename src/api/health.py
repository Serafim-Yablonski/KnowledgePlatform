import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(tags=["ops"])


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

    # Redis is the broker; use the module-level async client (same as the app uses).
    try:
        from src.core.redis import get_async_redis_client

        await get_async_redis_client().ping()
        checks["redis"] = "ok"
    except RedisError:
        checks["redis"] = "error"
    except RuntimeError:
        checks["redis"] = "unavailable"

    # Ping Celery workers via the broker. run_in_executor avoids blocking the
    # async handler — inspect() uses a synchronous kombu transport internally.
    try:
        from src.workers.celery_app import celery_app

        loop = asyncio.get_running_loop()
        ping_result: Any = await loop.run_in_executor(
            None,
            lambda: celery_app.control.inspect(timeout=1.0).ping(),
        )
        checks["workers"] = "ok" if ping_result else "unavailable"
    except Exception:
        checks["workers"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "healthy" if all_ok else "unhealthy", "checks": checks},
    )
