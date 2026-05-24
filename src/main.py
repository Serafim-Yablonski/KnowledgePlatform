import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.ai.graphs.checkpointer import setup_checkpointer
from src.api.health import router as health_router
from src.api.v1 import router as v1_router
from src.core.config import get_settings
from src.core.database import engine
from src.core.exceptions import AppError, RateLimitError
from src.core.logging import setup_logging
from src.core.observability import setup_observability
from src.core.redis import close_redis, init_redis

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    cfg = get_settings()

    setup_logging(environment=cfg.ENVIRONMENT)
    setup_observability(app, token=cfg.LOGFIRE_TOKEN or None)

    redis_client = await init_redis()
    app.state.engine = engine
    app.state.redis = redis_client

    async with setup_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        logger.info("startup complete", environment=cfg.ENVIRONMENT)
        yield

    await engine.dispose()
    await close_redis()
    logger.info("shutdown complete")


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    err = exc if isinstance(exc, AppError) else AppError()
    headers: dict[str, str] = {}
    if isinstance(err, RateLimitError):
        headers["Retry-After"] = str(err.retry_after)
    return JSONResponse(
        status_code=err.status_code,
        content={
            "error": {
                "code": err.status_code,
                "message": err.detail,
                "errors": err.errors,
            }
        },
        headers=headers or None,
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception",
        path=str(request.url),
        traceback=traceback.format_exception(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {"code": 500, "message": "Internal server error", "errors": None}
        },
    )


def create_app() -> FastAPI:
    application = FastAPI(
        title="Knowledge Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_exception_handler(AppError, _app_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)
    application.include_router(health_router)
    application.include_router(v1_router)

    from src.mcp_server.server import create_mcp_app

    application.mount("/mcp", create_mcp_app())

    return application


app: FastAPI = create_app()
