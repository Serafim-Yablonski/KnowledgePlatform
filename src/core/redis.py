from __future__ import annotations

import redis
import redis.asyncio as aioredis
from fastapi import Request

from src.core.config import settings

_PREFIX_CACHE = "nexus:cache"
_PREFIX_RATELIMIT = "nexus:ratelimit"

_async_pool: aioredis.ConnectionPool | None = None
_async_client: aioredis.Redis | None = None
_sync_client: redis.Redis | None = None


def _cache_key(key: str) -> str:
    return f"{_PREFIX_CACHE}:{key}"


# ---------------------------------------------------------------------------
# Lifecycle helpers — call from app lifespan
# ---------------------------------------------------------------------------


async def init_redis() -> aioredis.Redis:
    global _async_pool, _async_client
    _async_pool = aioredis.ConnectionPool.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        max_connections=20,
    )
    _async_client = aioredis.Redis(connection_pool=_async_pool)
    return _async_client


async def close_redis() -> None:
    global _async_pool, _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
    if _async_pool is not None:
        await _async_pool.aclose()
        _async_pool = None


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis  # type: ignore[no-any-return]


def get_async_redis_client() -> aioredis.Redis:
    """Return the module-level async Redis client (initialized by init_redis).

    Use this outside of FastAPI request context, e.g., in MCP tool functions.
    """
    if _async_client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    return _async_client


# ---------------------------------------------------------------------------
# Async cache helpers (FastAPI / coroutines)
# ---------------------------------------------------------------------------


async def cache_get(key: str) -> str | None:
    if _async_client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    val: str | None = await _async_client.get(_cache_key(key))
    return val


async def cache_set(key: str, value: str, ttl_seconds: int) -> None:
    if _async_client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    await _async_client.set(_cache_key(key), value, ex=ttl_seconds)


async def cache_delete(key: str) -> None:
    if _async_client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    await _async_client.delete(_cache_key(key))


# ---------------------------------------------------------------------------
# Sync cache helpers (Celery workers)
# ---------------------------------------------------------------------------


def _get_sync_client() -> redis.Redis:
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _sync_client


def sync_cache_get(key: str) -> str | None:
    # redis-py types sync get() as bytes|None even with decode_responses=True.
    val: str | None = _get_sync_client().get(_cache_key(key))  # type: ignore[assignment]
    return val


def sync_cache_set(key: str, value: str, ttl_seconds: int) -> None:
    _get_sync_client().set(_cache_key(key), value, ex=ttl_seconds)


def sync_cache_delete(key: str) -> None:
    _get_sync_client().delete(_cache_key(key))
