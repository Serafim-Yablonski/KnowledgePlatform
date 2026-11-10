"""Unit tests for ResponseCache's claim, the one operation it keeps (ADR-034)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from src.core.cache import ResponseCache


@pytest.fixture
def cache() -> ResponseCache:
    return ResponseCache(FakeRedis())


def _broken_redis() -> MagicMock:
    """Return a mock Redis client whose SET raises ConnectionError."""
    client = MagicMock()
    client.set = AsyncMock(side_effect=RedisConnectionError("Redis down"))
    return client


class TestClaim:
    @pytest.mark.asyncio
    async def test_first_claim_wins_until_ttl(self, cache: ResponseCache) -> None:
        assert await cache.claim("slot", ttl=60) is True
        assert await cache.claim("slot", ttl=60) is False

    @pytest.mark.asyncio
    async def test_claim_fails_closed_when_redis_down(self) -> None:
        # Nobody wins a slot Redis can't arbitrate: "everyone wins" would break
        # callers that use a claim for mutual exclusion.
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]
        with structlog.testing.capture_logs() as logs:
            assert await cache.claim("slot", ttl=60) is False
        assert any("cache claim failed" in e["event"] for e in logs)
