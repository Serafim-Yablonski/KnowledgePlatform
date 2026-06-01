from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

T = TypeVar("T")

logger = structlog.get_logger(__name__)

# Distinguishes "key absent in Redis" from "key present, value is JSON null".
_MISS: Any = object()


class ResponseCache:
    # Cache invalidation is the caller's responsibility. After any write that
    # modifies cached data, call `await self._cache.delete(key)` in the service
    # layer before or after committing. TTL is a safety net, not the primary
    # freshness mechanism. See service layer for workspace and API key invalidation.
    def __init__(
        self,
        redis_client: aioredis.Redis,
        key_prefix: str = "nexus:cache",
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    async def get(self, key: str) -> Any:
        try:
            raw: str | None = await self._redis.get(f"{self._prefix}:{key}")
        except RedisError as exc:
            logger.warning(
                "cache read failed — treating as miss", key=key, error=repr(exc)
            )
            return _MISS
        if raw is None:
            return _MISS
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self._redis.set(f"{self._prefix}:{key}", json.dumps(value), ex=ttl)
        except RedisError as exc:
            logger.warning(
                "cache write failed — value not cached", key=key, error=repr(exc)
            )

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(f"{self._prefix}:{key}")
        except RedisError as exc:
            logger.warning(
                "cache invalidation failed — stale entry will expire at TTL",
                key=key,
                error=repr(exc),
            )

    async def delete_pattern(self, pattern: str) -> int:
        full_pattern = f"{self._prefix}:{pattern}"
        deleted = 0
        cursor = 0
        try:
            while True:
                cursor, keys = await self._redis.scan(
                    cursor, match=full_pattern, count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
        except RedisError as exc:
            logger.warning(
                "cache pattern-delete failed — some stale entries may persist",
                pattern=full_pattern,
                error=repr(exc),
            )
        return deleted

    # Callers supply serializer/deserializer so Pydantic models
    # (and any other non-JSON-native type) round-trip correctly.
    # Defaults are identity functions — no change for primitive/dict callers.
    async def get_or_set(
        self,
        key: str,
        ttl: int,
        factory: Callable[[], Awaitable[T]],
        serializer: Callable[[T], Any] = lambda v: v,
        deserializer: Callable[[Any], T] = lambda v: v,
    ) -> T:
        cached = await self.get(key)
        if cached is not _MISS:
            return deserializer(cached)
        result = await factory()
        await self.set(key, serializer(result), ttl)
        return result


class CacheKeys:
    """Central registry of Redis key constructors.

    Each static method returns a fully-qualified key (without the ResponseCache
    prefix). ResponseCache prepends ``nexus:cache:`` at storage time.

    Using static methods with f-strings — rather than str.format() on a
    constant — avoids re-parsing the format string on every call.
    """

    @staticmethod
    def workspace(workspace_id: uuid.UUID) -> str:
        return f"workspace:{workspace_id}"

    @staticmethod
    def membership(workspace_id: uuid.UUID, user_id: uuid.UUID) -> str:
        return f"membership:{workspace_id}:{user_id}"

    @staticmethod
    def api_key(key_hash: str) -> str:
        return f"api_key:{key_hash}"

    # Glob pattern for delete_pattern() — used when deleting a workspace to
    # invalidate all membership cache entries for that workspace in one pass.
    @staticmethod
    def membership_pattern(workspace_id: uuid.UUID) -> str:
        return f"membership:{workspace_id}:*"
