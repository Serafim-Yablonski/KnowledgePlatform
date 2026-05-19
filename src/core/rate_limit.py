from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, Response

from src.core.exceptions import RateLimitError
from src.core.redis import get_redis
from src.models.user import User


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
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    async def check(self, identifier: str) -> RateLimitResult:
        key = f"nexus:ratelimit:{self._key_prefix}:{identifier}"
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
        results = await pipe.execute()

        count: int = results[2]
        oldest_entries: list[tuple[str, float]] = results[3]

        if count > self._max_requests:
            if oldest_entries:
                oldest_ts: float = oldest_entries[0][1]
                retry_after = max(1, int(oldest_ts + self._window_seconds - now) + 1)
            else:
                retry_after = self._window_seconds
            raise RateLimitError(retry_after=retry_after)

        reset_at = datetime.fromtimestamp(now + self._window_seconds, tz=UTC)
        return RateLimitResult(
            remaining=self._max_requests - count,
            reset_at=reset_at,
            limit=self._max_requests,
        )


def rate_limit(
    key_prefix: str,
    max_requests: int,
    window_seconds: int,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Return a FastAPI dependency function for rate limiting.

    Usage: Depends(rate_limit("search", 20, 60))
    """
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
            redis, key_prefix, max_requests, window_seconds
        )
        result = await limiter.check(str(user.id))
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at.timestamp()))

    return _check
