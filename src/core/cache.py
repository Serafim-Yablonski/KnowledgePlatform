from __future__ import annotations

import uuid

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

logger = structlog.get_logger(__name__)


class ResponseCache:
    """Short-lived Redis markers. It holds no data copies: authorization reads
    PostgreSQL (ADR-034)."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        key_prefix: str = "nexus:cache",
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    async def claim(self, key: str, ttl: int) -> bool:
        """Set ``key`` only if absent (SET NX); True means the caller won the slot.

        Fails closed: if Redis is down nobody wins, since a slot Redis can't arbitrate
        isn't exclusive. Callers that must still do their work handle False themselves.
        """
        try:
            return bool(
                await self._redis.set(f"{self._prefix}:{key}", 1, nx=True, ex=ttl)
            )
        except RedisError as exc:
            logger.warning("cache claim failed — skipping", key=key, error=repr(exc))
            return False


class CacheKeys:
    """Central registry of Redis key constructors.

    Each static method returns a key without the ResponseCache prefix, which
    ResponseCache prepends (``nexus:cache:``) at storage time.
    """

    @staticmethod
    def api_key_touched(key_id: uuid.UUID) -> str:
        return f"api_key_touched:{key_id}"
