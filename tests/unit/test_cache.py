"""Unit tests for ResponseCache."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing
from fakeredis.aioredis import FakeRedis
from pydantic import BaseModel
from redis.exceptions import ConnectionError as RedisConnectionError

from src.core.cache import ResponseCache


class _Item(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def cache(fake_redis: FakeRedis) -> ResponseCache:
    return ResponseCache(fake_redis)


# ---------------------------------------------------------------------------
# ResponseCache.get_or_set
# ---------------------------------------------------------------------------


class TestGetOrSet:
    @pytest.mark.asyncio
    async def test_calls_factory_on_miss(self, cache: ResponseCache) -> None:
        call_count = 0

        async def factory() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        result = await cache.get_or_set("k", 60, factory)

        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_does_not_call_factory_on_hit(self, cache: ResponseCache) -> None:
        call_count = 0

        async def factory() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        await cache.get_or_set("k", 60, factory)
        result = await cache.get_or_set("k", 60, factory)

        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_caches_none_result(self, cache: ResponseCache) -> None:
        # Bug 1 fix: a None return must be cached; previously the None check
        # conflated "key absent" with "stored null", causing re-calls every time.
        call_count = 0

        async def factory() -> None:
            nonlocal call_count
            call_count += 1
            return None

        result1 = await cache.get_or_set("k", 60, factory)
        result2 = await cache.get_or_set("k", 60, factory)

        assert result1 is None
        assert result2 is None
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_pydantic_round_trip(self, cache: ResponseCache) -> None:
        # Bug 2b fix: without serializer/deserializer, json.dumps(pydantic_model)
        # would raise TypeError. With them, the model survives the Redis round-trip.
        original = _Item(name="bar", value=7)

        async def factory() -> _Item:
            return original

        result1 = await cache.get_or_set(
            "item",
            60,
            factory,
            serializer=lambda r: r.model_dump(mode="json"),
            deserializer=_Item.model_validate,
        )
        result2 = await cache.get_or_set(
            "item",
            60,
            factory,
            serializer=lambda r: r.model_dump(mode="json"),
            deserializer=_Item.model_validate,
        )

        assert result1 == original
        assert isinstance(result1, _Item)
        assert result2 == original
        # Cache hit must return an _Item instance, not a raw dict.
        assert isinstance(result2, _Item)


# ---------------------------------------------------------------------------
# ResponseCache.delete
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_causes_next_call_to_miss(self, cache: ResponseCache) -> None:
        call_count = 0

        async def factory() -> int:
            nonlocal call_count
            call_count += 1
            return 1

        await cache.get_or_set("k", 60, factory)
        await cache.delete("k")
        await cache.get_or_set("k", 60, factory)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key_is_safe(self, cache: ResponseCache) -> None:
        await cache.delete("does-not-exist")  # must not raise


# ---------------------------------------------------------------------------
# Graceful degradation when Redis is unavailable
# ---------------------------------------------------------------------------


def _broken_redis() -> MagicMock:
    """Return a mock Redis client whose async commands all raise ConnectionError."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=RedisConnectionError("Redis down"))
    client.set = AsyncMock(side_effect=RedisConnectionError("Redis down"))
    client.delete = AsyncMock(side_effect=RedisConnectionError("Redis down"))
    client.scan = AsyncMock(side_effect=RedisConnectionError("Redis down"))
    return client


class TestRedisDownDegradation:
    @pytest.mark.asyncio
    async def test_get_returns_miss_and_logs_warning(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]

        with structlog.testing.capture_logs() as logs:
            await cache.get("some-key")

        warning_events = [e for e in logs if e["log_level"] == "warning"]
        assert any("cache read failed" in e["event"] for e in warning_events)

    @pytest.mark.asyncio
    async def test_get_does_not_raise(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]
        with structlog.testing.capture_logs():
            await cache.get("k")  # must not raise

    @pytest.mark.asyncio
    async def test_set_does_not_raise_and_logs_warning(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]

        with structlog.testing.capture_logs() as logs:
            await cache.set("k", {"v": 1}, ttl=60)

        warning_events = [e for e in logs if e["log_level"] == "warning"]
        assert any("cache write failed" in e["event"] for e in warning_events)

    @pytest.mark.asyncio
    async def test_delete_does_not_raise_and_logs_warning(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]

        with structlog.testing.capture_logs() as logs:
            await cache.delete("k")

        warning_events = [e for e in logs if e["log_level"] == "warning"]
        assert any("cache invalidation failed" in e["event"] for e in warning_events)

    @pytest.mark.asyncio
    async def test_delete_pattern_returns_zero_and_logs_warning(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]

        with structlog.testing.capture_logs() as logs:
            deleted = await cache.delete_pattern("workspace:*")

        assert deleted == 0
        warning_events = [e for e in logs if e["log_level"] == "warning"]
        assert any("cache pattern-delete failed" in e["event"] for e in warning_events)

    @pytest.mark.asyncio
    async def test_get_or_set_calls_factory_when_redis_down(self) -> None:
        cache = ResponseCache(_broken_redis())  # type: ignore[arg-type]
        call_count = 0

        async def factory() -> int:
            nonlocal call_count
            call_count += 1
            return 99

        with structlog.testing.capture_logs():
            result = await cache.get_or_set("k", 60, factory)

        assert result == 99
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_set_returns_factory_result_even_if_set_fails(self) -> None:
        # Redis is up for get() (returns None → miss) but fails on set().
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(side_effect=RedisConnectionError("Redis down"))
        cache = ResponseCache(client)  # type: ignore[arg-type]

        with structlog.testing.capture_logs():
            result = await cache.get_or_set("k", 60, lambda: _async_const(42))

        assert result == 42


async def _async_const(value: int) -> int:
    return value
