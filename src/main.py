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
from src.core.http import close_http_client, init_http_client
from src.core.logging import setup_logging
from src.core.observability import setup_observability
from src.core.redis import close_redis, init_redis

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    cfg = get_settings()

    setup_logging(environment=cfg.ENVIRONMENT, log_level=cfg.LOG_LEVEL)
    setup_observability(
        app,
        token=cfg.LOGFIRE_TOKEN or None,
        environment=cfg.ENVIRONMENT,
        service_name=cfg.SERVICE_NAME,
    )

    await init_redis()
    await init_http_client()
    app.state.engine = engine

    from src.mcp_server.server import get_mcp_session_manager

    async with setup_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        # StreamableHTTPSessionManager requires an anyio task group; mount()
        # does not invoke sub-app lifespans, so we start it here explicitly.
        async with get_mcp_session_manager().run():
            logger.info("startup complete", environment=cfg.ENVIRONMENT)
            yield

    await close_http_client()
    await engine.dispose()
    await close_redis()
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
    application.add_exception_handler(AppError, _app_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)
    application.include_router(health_router)
    application.include_router(v1_router)

    cfg = get_settings()
    if cfg.TRUSTED_PROXY_HEADERS:
        from starlette.middleware.proxy_headers import ProxyHeadersMiddleware  # type: ignore[import-not-found]  # noqa: PLC0415,E501,I001

        trusted: list[str] | str = (
            "*"
            if cfg.TRUSTED_PROXY_IPS.strip() == "*"
            else [ip.strip() for ip in cfg.TRUSTED_PROXY_IPS.split(",") if ip.strip()]
        )
        application.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted)

    from src.mcp_server.server import create_mcp_app

    # Mount at "/" so FastMCP's inner route "/mcp" becomes "http://host/mcp".
    # Explicit routes (health, v1) added above via include_router take priority.
    application.mount("/", create_mcp_app())

    return application


app: FastAPI = create_app()
