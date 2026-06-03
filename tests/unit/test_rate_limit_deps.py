"""Tests for rate-limit dependency wrappers and SlidingWindowRateLimiter.fail_open."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from src.core.exceptions import AppError, RateLimitError, ServiceUnavailableError
from src.core.rate_limit import (
    SlidingWindowRateLimiter,
    ip_rate_limit,
    rate_limit,
    workspace_rate_limit,
)
from src.core.redis import get_redis


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    err = exc if isinstance(exc, AppError) else AppError()
    headers: dict[str, str] = {}
    if isinstance(err, RateLimitError):
        headers["Retry-After"] = str(err.retry_after)
        headers["RateLimit-Limit"] = str(err.limit)
        headers["RateLimit-Remaining"] = "0"
        headers["RateLimit-Reset"] = str(err.retry_after)
    return JSONResponse(
        status_code=err.status_code,
        content={"error": {"code": err.status_code, "message": err.detail}},
        headers=headers or None,
    )


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(AppError, _app_error_handler)
    return app


def _make_redis(count: int, oldest_ts: float | None = None) -> MagicMock:
    oldest = [(b"member", oldest_ts)] if oldest_ts is not None else []
    pipe = MagicMock()
    pipe.zadd = MagicMock()
    pipe.zremrangebyscore = MagicMock()
    pipe.zcard = MagicMock()
    pipe.zrange = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[None, None, count, oldest, True])
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    redis.zrem = AsyncMock()
    return redis


def _make_redis_error() -> MagicMock:
    pipe = MagicMock()
    for method in ("zadd", "zremrangebyscore", "zcard", "zrange", "expire"):
        setattr(pipe, method, MagicMock())
    pipe.execute = AsyncMock(side_effect=RedisError("down"))
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    return redis


# ---------------------------------------------------------------------------
# fail_open behaviour
# ---------------------------------------------------------------------------


class TestFailOpen:
    @pytest.mark.asyncio
    async def test_fail_open_true_allows_request(self) -> None:
        limiter = SlidingWindowRateLimiter(
            _make_redis_error(), "test", 5, 60, fail_open=True
        )
        result = await limiter.check("user-1")
        assert result.remaining == 5

    @pytest.mark.asyncio
    async def test_fail_open_false_raises_503(self) -> None:
        limiter = SlidingWindowRateLimiter(
            _make_redis_error(), "test", 5, 60, fail_open=False
        )
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await limiter.check("user-1")
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Rejection logging
# ---------------------------------------------------------------------------


class TestRejectionLogging:
    @pytest.mark.asyncio
    async def test_rejection_emits_warning(self) -> None:
        redis = _make_redis(count=6, oldest_ts=time.time() - 10)
        limiter = SlidingWindowRateLimiter(redis, "search", 5, 60)
        with (
            patch("src.core.rate_limit.logger") as mock_logger,
            pytest.raises(RateLimitError),
        ):
            await limiter.check("user-abc")
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args.args[0] == "rate_limit_exceeded"
        assert call_args.kwargs["key_prefix"] == "search"
        assert call_args.kwargs["identifier"] == "user-abc"


# ---------------------------------------------------------------------------
# ip_rate_limit dependency — headers and 429
# ---------------------------------------------------------------------------


def _build_ip_app(count: int) -> tuple[FastAPI, MagicMock]:
    app = _make_app()
    mock_redis = _make_redis(count)
    app.dependency_overrides[get_redis] = lambda: mock_redis

    dep = ip_rate_limit("test_ip", 5, 60)

    @app.get("/test", dependencies=[__import__("fastapi").Depends(dep)])
    async def _route() -> dict:
        return {"ok": True}

    return app, mock_redis


class TestIpRateLimitDep:
    def test_sets_ratelimit_headers(self) -> None:
        app, _ = _build_ip_app(count=1)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.headers["RateLimit-Limit"] == "5"
        assert "RateLimit-Remaining" in resp.headers
        assert "RateLimit-Reset" in resp.headers

    def test_returns_429_when_over_limit(self) -> None:
        app, _ = _build_ip_app(count=6)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert resp.headers["RateLimit-Limit"] == "5"
        assert resp.headers["RateLimit-Remaining"] == "0"
        assert "RateLimit-Reset" in resp.headers

    def test_skips_rate_limit_when_client_ip_unknown(self) -> None:
        """Requests with no resolvable IP must pass through without sharing a bucket."""
        app = _make_app()
        mock_redis = _make_redis(count=999)  # would trip any limit if checked
        app.dependency_overrides[get_redis] = lambda: mock_redis

        dep = ip_rate_limit("test_ip", 5, 60)

        @app.get("/test", dependencies=[__import__("fastapi").Depends(dep)])
        async def _route() -> dict:
            return {"ok": True}

        # Patch _client_ip to simulate a None-client environment (UNIX socket, etc.)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            patch("src.core.rate_limit._client_ip", return_value=None),
        ):
            resp = client.get("/test")

        assert resp.status_code == 200
        assert "RateLimit-Limit" not in resp.headers
        mock_redis.pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# rate_limit dependency — user-scoped key
# ---------------------------------------------------------------------------


def _build_user_app(count: int) -> tuple[FastAPI, MagicMock]:
    from src.core.dependencies import get_current_user

    app = _make_app()
    mock_redis = _make_redis(count)
    mock_user = MagicMock()
    mock_user.id = "user-uuid-123"
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_current_user] = lambda: mock_user

    dep = rate_limit("test_user", 5, 60)

    @app.get("/test", dependencies=[__import__("fastapi").Depends(dep)])
    async def _route() -> dict:
        return {"ok": True}

    return app, mock_redis


class TestRateLimitDep:
    def test_sets_ratelimit_headers(self) -> None:
        app, _ = _build_user_app(count=1)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.headers["RateLimit-Limit"] == "5"

    def test_returns_429_when_over_limit(self) -> None:
        app, _ = _build_user_app(count=6)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 429
        assert resp.headers["RateLimit-Limit"] == "5"
        assert resp.headers["RateLimit-Remaining"] == "0"
        assert "RateLimit-Reset" in resp.headers

    def test_uses_user_id_as_key(self) -> None:
        """Redis key must contain the user id, not a shared key."""
        app, mock_redis = _build_user_app(count=1)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.get("/test")
        zadd_key = mock_redis.pipeline.return_value.zadd.call_args.args[0]
        assert "user-uuid-123" in zadd_key


# ---------------------------------------------------------------------------
# workspace_rate_limit dependency — (user_id, workspace_id) key
# ---------------------------------------------------------------------------


def _build_workspace_app(count: int) -> tuple[FastAPI, MagicMock]:
    from src.core.dependencies import get_current_user, get_current_workspace

    app = _make_app()
    mock_redis = _make_redis(count)
    mock_user = MagicMock()
    mock_user.id = "user-uuid-456"
    mock_workspace = MagicMock()
    mock_workspace.id = "ws-uuid-789"
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_current_workspace] = lambda: mock_workspace

    dep = workspace_rate_limit("test_ws", 5, 60)

    @app.get("/test", dependencies=[__import__("fastapi").Depends(dep)])
    async def _route() -> dict:
        return {"ok": True}

    return app, mock_redis


class TestWorkspaceRateLimitDep:
    def test_sets_ratelimit_headers(self) -> None:
        app, _ = _build_workspace_app(count=1)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.headers["RateLimit-Limit"] == "5"

    def test_returns_429_when_over_limit(self) -> None:
        app, _ = _build_workspace_app(count=6)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 429
        assert resp.headers["RateLimit-Limit"] == "5"
        assert resp.headers["RateLimit-Remaining"] == "0"
        assert "RateLimit-Reset" in resp.headers

    def test_uses_composite_key(self) -> None:
        """Redis key must contain both user_id and workspace_id."""
        app, mock_redis = _build_workspace_app(count=1)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.get("/test")
        zadd_key = mock_redis.pipeline.return_value.zadd.call_args.args[0]
        assert "user-uuid-456" in zadd_key
        assert "ws-uuid-789" in zadd_key
