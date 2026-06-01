from __future__ import annotations

import redis
import redis.asyncio as aioredis

from src.core.config import settings

# Problem 5 fix: _PREFIX_CACHE and the standalone cache_get/set/delete helpers have been
# removed — they were unused and duplicated ResponseCache (cache.py) with a different
# prefix and no JSON serialization, making cross-system invalidation impossible.
PREFIX_RATELIMIT = "nexus:ratelimit"

_async_pool: aioredis.ConnectionPool | None = None
_async_client: aioredis.Redis | None = None
_sync_client: redis.Redis | None = None


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


def get_redis() -> aioredis.Redis:
    # Problem 8 fix: single source of truth — the module-level client set by
    # init_redis(). Previously read from request.app.state.redis, which worked
    # only because main.py happened to store the same object there; that
    # implicit coupling is now removed.
    return get_async_redis_client()


def get_async_redis_client() -> aioredis.Redis:
    """Return the module-level async Redis client (initialized by init_redis).

    Use this outside of FastAPI request context, e.g., in MCP tool functions.
    """
    if _async_client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    return _async_client


# ---------------------------------------------------------------------------
# Sync client — Celery workers only
# ---------------------------------------------------------------------------


def _get_sync_client() -> redis.Redis:
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _sync_client


def close_sync_redis() -> None:
    # Problem 7 fix: release sync connections on Celery worker shutdown.
    # Called from the worker_shutdown signal in src/workers/celery_app.py.
    global _sync_client
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
