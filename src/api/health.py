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

    redis_client: Any = getattr(request.app.state, "redis", None)
    if redis_client is None:
        checks["redis"] = "unavailable"
    else:
        try:
            await redis_client.ping()
            checks["redis"] = "ok"
        except RedisError:
            checks["redis"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "healthy" if all_ok else "unhealthy", "checks": checks},
    )
