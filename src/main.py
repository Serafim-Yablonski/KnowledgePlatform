import traceback
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypedDict

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.routing import Route
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from src.ai.graphs.checkpointer import setup_checkpointer
from src.api.health import router as health_router
from src.api.v1 import router as v1_router
from src.composition import Services, build_services
from src.core.config import get_settings
from src.core.database import build_engine
from src.core.exceptions import (
    AppError,
    RateLimitError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from src.core.http import build_http_client
from src.core.logging import setup_logging
from src.core.observability import instrument_app, setup_observability
from src.core.redis import build_redis

logger = structlog.get_logger(__name__)


class LifespanState(TypedDict):
    """What the lifespan hands every request (Starlette's lifespan state: the server
    copies it into each request's ``request.state``). REST reads it in
    ``src.core.dependencies`` and ``src.api.health``, MCP in ``src.mcp_server.auth``.
    """

    services: Services
    engine: AsyncEngine  # /health's own probe
    redis: aioredis.Redis  # the rate limiters and /health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[LifespanState]:
    """The process's pools and clients, opened here and nowhere at import, so each
    app instance has its own (ADR-033)."""
    cfg = get_settings()

    setup_logging(environment=cfg.ENVIRONMENT, log_level=cfg.LOG_LEVEL)
    setup_observability(
        token=cfg.LOGFIRE_TOKEN.get_secret_value() or None,
        environment=cfg.ENVIRONMENT,
        service_name=cfg.SERVICE_NAME,
    )

    async with AsyncExitStack() as stack:
        engine = build_engine(cfg.ASYNC_DATABASE_URL.get_secret_value())
        stack.push_async_callback(engine.dispose)
        redis = await stack.enter_async_context(
            build_redis(cfg.REDIS_URL.get_secret_value())
        )
        http = await stack.enter_async_context(build_http_client())
        await stack.enter_async_context(setup_checkpointer())
        # The MCP session manager handles requests in an anyio task group that
        # only its run() provides; nothing else starts it.
        await stack.enter_async_context(app.state.mcp_session_manager.run())
        logger.info("startup complete", environment=cfg.ENVIRONMENT)
        yield LifespanState(
            services=build_services(engine, redis, http), engine=engine, redis=redis
        )
    logger.info("shutdown complete")


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    err = exc if isinstance(exc, AppError) else AppError()
    log_fn = logger.error if err.status_code >= 500 else logger.warning
    log_fn(
        "request error",
        error_type=type(err).__name__,
        path=str(request.url.path),
        method=request.method,
        status_code=err.status_code,
    )
    headers: dict[str, str] = {}
    if isinstance(err, RateLimitError):
        headers["Retry-After"] = str(err.retry_after)
        headers["RateLimit-Limit"] = str(err.limit)
        headers["RateLimit-Remaining"] = "0"
        headers["RateLimit-Reset"] = str(err.retry_after)
    if isinstance(err, ServiceUnavailableError) and err.retry_after is not None:
        headers["Retry-After"] = str(err.retry_after)
    if isinstance(err, UnauthorizedError):
        headers["WWW-Authenticate"] = err.www_authenticate
    # Only surface structured field errors on validation responses; strip the
    # free-form errors dict from all other codes to avoid leaking internals.
    errors = err.errors if err.status_code == 422 else None
    return JSONResponse(
        status_code=err.status_code,
        content={
            "error": {
                "code": err.status_code,
                "message": err.detail,
                "errors": errors,
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
    instrument_app(application)
    application.add_exception_handler(AppError, _app_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)
    application.include_router(health_router)
    application.include_router(v1_router)

    cfg = get_settings()
    if cfg.TRUSTED_PROXY_HEADERS:
        trusted: list[str] | str = (
            "*"
            if cfg.TRUSTED_PROXY_IPS.strip() == "*"
            else [ip.strip() for ip in cfg.TRUSTED_PROXY_IPS.split(",") if ip.strip()]
        )
        application.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted)

    from src.mcp_server.server import create_mcp_app

    # An exact route, not a mount: a Mount("/mcp") answers POST /mcp with a 307 to
    # /mcp/, and a Mount("/") would catch every unmatched REST path with MCP's 401.
    # A non-function endpoint is served as a raw ASGI app, every method included.
    mcp_app, application.state.mcp_session_manager = create_mcp_app()
    application.router.routes.append(Route("/mcp", mcp_app))

    return application


app: FastAPI = create_app()
