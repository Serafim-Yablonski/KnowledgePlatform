from __future__ import annotations

import redis.asyncio as aioredis

PREFIX_RATELIMIT = "nexus:ratelimit"


def build_redis(url: str) -> aioredis.Redis:
    """The API process's Redis client; its lifespan builds one (ADR-033)."""
    client: aioredis.Redis = aioredis.Redis.from_url(
        url, decode_responses=True, max_connections=20
    )
    return client
