"""Unit tests for SlidingWindowRateLimiter."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from src.core.exceptions import RateLimitError
from src.core.rate_limit import RateLimitResult, SlidingWindowRateLimiter, _client_ip


def _make_redis(pipeline_results: list) -> MagicMock:
    """Build a mock Redis client whose pipeline returns the given results.

    Pipeline command methods (zadd, zcard, etc.) are sync in redis-py — they
    just enqueue commands.  Only execute() is async.
    """
    pipe = MagicMock()
    pipe.zadd = MagicMock()
    pipe.zremrangebyscore = MagicMock()
    pipe.zcard = MagicMock()
    pipe.zrange = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=pipeline_results)

    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    redis.zrem = AsyncMock()
    return redis


def _results(count: int, oldest_ts: float | None = None) -> list:
    """Build the 5-element pipeline result list."""
    oldest = [(b"member", oldest_ts)] if oldest_ts is not None else []
    return [None, None, count, oldest, True]


class TestSlidingWindowRateLimiter:
    def _make(
        self, max_requests: int = 20, window_seconds: int = 60
    ) -> tuple[SlidingWindowRateLimiter, MagicMock]:
        redis = _make_redis([])
        limiter = SlidingWindowRateLimiter(redis, "test", max_requests, window_seconds)
        return limiter, redis

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_first_request_succeeds(self) -> None:
        limiter, redis = self._make(max_requests=20)
        redis.pipeline.return_value.execute = AsyncMock(return_value=_results(1))

        result = await limiter.check("user-1")

        assert isinstance(result, RateLimitResult)
        assert result.remaining == 19
        assert result.limit == 20

    @pytest.mark.asyncio
    async def test_exactly_at_limit_succeeds(self) -> None:
        limiter, redis = self._make(max_requests=20)
        redis.pipeline.return_value.execute = AsyncMock(return_value=_results(20))

        result = await limiter.check("user-1")

        assert result.remaining == 0

    # ------------------------------------------------------------------
    # Rate limit exceeded
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_over_limit_raises(self) -> None:
        limiter, redis = self._make(max_requests=20)
        now = time.time()
        oldest_ts = now - 10  # 10 seconds ago → 50 seconds until expiry
        redis.pipeline.return_value.execute = AsyncMock(
            return_value=_results(21, oldest_ts=oldest_ts)
        )

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.check("user-1")

        assert exc_info.value.retry_after >= 1

    @pytest.mark.asyncio
    async def test_retry_after_reflects_oldest_entry(self) -> None:
        """retry_after should be ~window - age_of_oldest_entry."""
        limiter, redis = self._make(max_requests=5, window_seconds=60)
        now = time.time()
        oldest_ts = now - 55  # oldest request was 55 s ago → expires in ~5 s
        redis.pipeline.return_value.execute = AsyncMock(
            return_value=_results(6, oldest_ts=oldest_ts)
        )

        with patch("src.core.rate_limit.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.time_ns.return_value = int(now * 1e9)
            with pytest.raises(RateLimitError) as exc_info:
                await limiter.check("user-1")

        # Should be 60 - 55 + 1 = 6, give or take 1 for int rounding
        assert 5 <= exc_info.value.retry_after <= 7

    @pytest.mark.asyncio
    async def test_over_limit_no_oldest_entry_falls_back(self) -> None:
        """When the sorted set is empty (race condition), use full window."""
        limiter, redis = self._make(max_requests=5, window_seconds=60)
        results = [None, None, 6, [], True]
        redis.pipeline.return_value.execute = AsyncMock(return_value=results)

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.check("user-1")

        assert exc_info.value.retry_after == 60

    # ------------------------------------------------------------------
    # User isolation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_different_users_use_different_keys(self) -> None:
        redis = _make_redis(_results(1))
        limiter = SlidingWindowRateLimiter(redis, "search", 20, 60)

        await limiter.check("user-a")
        await limiter.check("user-b")

        calls = redis.pipeline.return_value.zadd.call_args_list
        keys = [c.args[0] for c in calls]
        assert keys[0] != keys[1]
        assert "user-a" in keys[0]
        assert "user-b" in keys[1]

    # ------------------------------------------------------------------
    # Pipeline is used (single round trip)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_uses_pipeline(self) -> None:
        limiter, redis = self._make()
        redis.pipeline.return_value.execute = AsyncMock(return_value=_results(1))

        await limiter.check("user-1")

        redis.pipeline.assert_called_once_with(transaction=True)
        redis.pipeline.return_value.execute.assert_awaited_once()

    # ------------------------------------------------------------------
    # Redis unavailable → fail open
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_redis_error_fails_open(self) -> None:
        from redis.exceptions import RedisError

        limiter, redis = self._make(max_requests=5)
        redis.pipeline.return_value.execute = AsyncMock(side_effect=RedisError("down"))

        result = await limiter.check("user-1")

        assert result.remaining == 5
        assert result.limit == 5

    # ------------------------------------------------------------------
    # Rejected requests are removed from the window
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rejected_request_removed_from_window(self) -> None:
        limiter, redis = self._make(max_requests=5)
        redis.pipeline.return_value.execute = AsyncMock(
            return_value=_results(6, oldest_ts=time.time() - 10)
        )
        redis.zrem = AsyncMock()

        with pytest.raises(RateLimitError):
            await limiter.check("user-1")

        redis.zrem.assert_awaited_once()


class TestResetAt:
    @pytest.mark.asyncio
    async def test_reset_at_uses_oldest_entry_expiry(self) -> None:
        """reset_at should reflect when the oldest entry expires, not now+window."""
        redis = _make_redis([])
        limiter = SlidingWindowRateLimiter(redis, "test", 5, 60)
        now = time.time()
        oldest_ts = now - 40  # oldest request was 40s ago → expires in 20s
        redis.pipeline.return_value.execute = AsyncMock(
            return_value=_results(3, oldest_ts=oldest_ts)
        )

        with patch("src.core.rate_limit.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.time_ns.return_value = int(now * 1e9)
            result = await limiter.check("user-1")

        expected_reset = oldest_ts + 60
        assert abs(result.reset_at.timestamp() - expected_reset) < 1

    @pytest.mark.asyncio
    async def test_reset_at_falls_back_when_no_entries(self) -> None:
        """When the zset is empty (first-ever request race), fall back to now+window."""
        redis = _make_redis([])
        limiter = SlidingWindowRateLimiter(redis, "test", 5, 60)
        now = time.time()
        # No oldest entry (empty list)
        redis.pipeline.return_value.execute = AsyncMock(
            return_value=[None, None, 1, [], True]
        )

        with patch("src.core.rate_limit.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.time_ns.return_value = int(now * 1e9)
            result = await limiter.check("user-1")

        assert abs(result.reset_at.timestamp() - (now + 60)) < 1


class TestClientIp:
    def _req(self, host: str | None) -> Request:
        scope: dict = {"type": "http", "headers": []}
        if host is not None:
            scope["client"] = (host, 12345)
        req = Request(scope)
        return req

    def test_returns_host_when_client_present(self) -> None:
        assert _client_ip(self._req("1.2.3.4")) == "1.2.3.4"

    def test_returns_none_when_client_absent(self) -> None:
        assert _client_ip(self._req(None)) is None
