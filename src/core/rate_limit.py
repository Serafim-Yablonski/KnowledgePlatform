from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import logfire
import structlog
from fastapi import Depends, Request, Response
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.exceptions import RateLimitError, ServiceUnavailableError
from src.core.redis import PREFIX_RATELIMIT, get_redis
from src.models.user import User

logger = structlog.get_logger(__name__)

_rejection_counter = logfire.metric_counter(
    "rate_limit.rejected",
    unit="{request}",
    description="Requests rejected by rate limiting",
)


@dataclass
class RateLimitResult:
    remaining: int
    reset_at: datetime
    limit: int


class SlidingWindowRateLimiter:
    def __init__(
        self,
        redis_client: Any,  # redis.asyncio.Redis
        key_prefix: str,
        max_requests: int,
        window_seconds: int,
        *,
        fail_open: bool = True,
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._fail_open = fail_open

    async def check(self, identifier: str) -> RateLimitResult:
        key = f"{PREFIX_RATELIMIT}:{self._key_prefix}:{identifier}"
        now = time.time()
        window_start = now - self._window_seconds
        # Unique member per request: nanosecond timestamp avoids overwriting
        # concurrent requests that arrive within the same millisecond.
        member = str(time.time_ns())

        pipe = self._redis.pipeline(transaction=True)
        pipe.zadd(key, {member: now})
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zcard(key)
        pipe.zrange(key, 0, 0, withscores=True)  # oldest entry for retry_after
        pipe.expire(key, self._window_seconds + 1)
        try:
            results = await pipe.execute()
        except RedisError:
            logger.warning("rate_limit_redis_unavailable", key_prefix=self._key_prefix)
            if not self._fail_open:
                raise ServiceUnavailableError(  # noqa: B904
                    "Rate limit backend unavailable"
                ) from None
            return RateLimitResult(
                remaining=self._max_requests,
                reset_at=datetime.fromtimestamp(now + self._window_seconds, tz=UTC),
                limit=self._max_requests,
            )

        count: int = results[2]
        oldest_entries: list[tuple[str, float]] = results[3]

        if count > self._max_requests:
            # Remove the just-added member so a rejected request doesn't inflate
            # the window and permanently lock out the caller.
            # Note: this zrem is outside the pipeline — if Redis fails here the
            # count stays +1 for the window duration (one false-positive rejection
            # max). Acceptable tradeoff vs. adding a Lua script for atomicity.
            await self._redis.zrem(key, member)
            if oldest_entries:
                oldest_ts: float = oldest_entries[0][1]
                retry_after = max(1, int(oldest_ts + self._window_seconds - now) + 1)
            else:
                retry_after = self._window_seconds
            logger.warning(
                "rate_limit_exceeded",
                key_prefix=self._key_prefix,
                identifier=identifier,
                retry_after=retry_after,
            )
            _rejection_counter.add(1, {"key_prefix": self._key_prefix})
            raise RateLimitError(retry_after=retry_after, limit=self._max_requests)

        # Use the oldest entry's expiry as the reset time: if the oldest request
        # is 50s into a 60s window, the bucket has capacity again in ~10s, not 60s.
        reset_at = (
            datetime.fromtimestamp(oldest_entries[0][1] + self._window_seconds, tz=UTC)
            if oldest_entries
            else datetime.fromtimestamp(now + self._window_seconds, tz=UTC)
        )
        return RateLimitResult(
            remaining=self._max_requests - count,
            reset_at=reset_at,
            limit=self._max_requests,
        )


def _client_ip(request: Request) -> str | None:
    # When TRUSTED_PROXY_HEADERS=True, ProxyHeadersMiddleware (configured in
    # create_app) has already rewritten request.client from the X-Forwarded-For
    # header before this function runs. No header parsing here — trust the ASGI layer.
    # Returns None for UNIX socket transports where the ASGI scope has no client.
    return request.client.host if request.client else None


def _set_headers(response: Response, result: RateLimitResult, limit: int) -> None:
    now = datetime.now(tz=UTC)
    reset_seconds = max(0, int((result.reset_at - now).total_seconds()))
    response.headers["RateLimit-Limit"] = str(limit)
    response.headers["RateLimit-Remaining"] = str(result.remaining)
    response.headers["RateLimit-Reset"] = str(reset_seconds)


def ip_rate_limit(
    key_prefix: str,
    max_requests: int,
    window_seconds: int,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """FastAPI dependency for IP-based rate limiting (unauthenticated routes)."""

    async def _check(
        request: Request,
        response: Response,
        redis: Any = Depends(get_redis),
    ) -> None:
        ip = _client_ip(request)
        if ip is None:
            logger.warning("rate_limit_ip_unknown", path=str(request.url.path))
            return  # cannot identify caller; skip rather than share a bucket
        limiter = SlidingWindowRateLimiter(
            redis,
            key_prefix,
            max_requests,
            window_seconds,
            fail_open=settings.RATE_LIMIT_FAIL_OPEN,
        )
        result = await limiter.check(ip)
        _set_headers(response, result, max_requests)

    return _check


def rate_limit(
    key_prefix: str,
    max_requests: int,
    window_seconds: int,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """FastAPI dependency for user-scoped rate limiting (authenticated routes)."""
    # Import here to avoid circular imports at module load time:
    # rate_limit.py is imported by api/v1/search.py, which is loaded after
    # dependencies.py, so this lazy import is safe.
    from src.core.dependencies import get_current_user  # noqa: PLC0415

    async def _check(
        response: Response,
        user: User = Depends(get_current_user),
        redis: Any = Depends(get_redis),
    ) -> None:
        limiter = SlidingWindowRateLimiter(
            redis,
            key_prefix,
            max_requests,
            window_seconds,
            fail_open=settings.RATE_LIMIT_FAIL_OPEN,
        )
        result = await limiter.check(str(user.id))
        _set_headers(response, result, max_requests)

    return _check


def workspace_rate_limit(
    key_prefix: str,
    max_requests: int,
    window_seconds: int,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """FastAPI dependency keyed on (user_id, workspace_id).

    Use for endpoints where resource cost is workspace-level (e.g. document
    upload triggers Celery tasks shared across all workspace members).
    """
    from src.core.dependencies import (  # noqa: PLC0415
        get_current_user,
        get_current_workspace,
    )

    async def _check(
        response: Response,
        user: User = Depends(get_current_user),
        workspace: Any = Depends(get_current_workspace),
        redis: Any = Depends(get_redis),
    ) -> None:
        limiter = SlidingWindowRateLimiter(
            redis,
            key_prefix,
            max_requests,
            window_seconds,
            fail_open=settings.RATE_LIMIT_FAIL_OPEN,
        )
        result = await limiter.check(f"{user.id}:{workspace.id}")
        _set_headers(response, result, max_requests)

    return _check
