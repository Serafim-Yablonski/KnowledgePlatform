from __future__ import annotations

import functools
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast, get_type_hints

import redis.asyncio as aioredis

T = TypeVar("T")


class ResponseCache:
    def __init__(
        self,
        redis_client: aioredis.Redis,
        key_prefix: str = "nexus:response",
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    async def get(self, key: str) -> Any | None:
        raw: str | None = await self._redis.get(f"{self._prefix}:{key}")
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        await self._redis.set(f"{self._prefix}:{key}", json.dumps(value), ex=ttl)

    async def delete_pattern(self, pattern: str) -> int:
        full_pattern = f"{self._prefix}:{pattern}"
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=full_pattern, count=100)
            if keys:
                await self._redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        return deleted

    async def get_or_set(
        self,
        key: str,
        ttl: int,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        cached = await self.get(key)
        if cached is not None:
            return cast(T, cached)
        result = await factory()
        await self.set(key, result, ttl)
        return result


def cached(ttl: int, key_template: str) -> Callable[..., Any]:
    """Decorator for async service methods.  The decorated method's instance
    must expose ``self._cache: ResponseCache``.

    Template variables are filled from the bound method arguments.
    Any ``{name_hash}`` variable is auto-computed as SHA-256[:16] of the
    argument named ``name``.

    Pydantic BaseModel return types are serialised via ``model_dump(mode='json')``
    and reconstructed with ``model_validate()`` on cache hit.

    Example::
        @cached(ttl=300, key_template="search:{workspace_id}:{query_hash}")
        async def search(self, workspace_id, query, ...): ...
    """

    def decorator(method: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        sig = inspect.signature(method)
        try:
            hints = get_type_hints(method)
        except (NameError, AttributeError):  # fmt: skip
            hints = {}
        return_type = hints.get("return")

        @functools.wraps(method)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            params: dict[str, Any] = {
                k: v for k, v in bound.arguments.items() if k != "self"
            }
            for name, value in list(params.items()):
                hash_key = f"{name}_hash"
                if f"{{{hash_key}}}" in key_template:
                    digest = hashlib.sha256(str(value).encode()).hexdigest()
                    params[hash_key] = digest[:16]

            key = key_template.format_map(params)

            raw = await self._cache.get(key)
            if raw is not None:
                if return_type is not None and hasattr(return_type, "model_validate"):
                    return cast(T, return_type.model_validate(raw))
                return cast(T, raw)

            result: T = await method(self, *args, **kwargs)

            # Serialize Pydantic models to a JSON-compatible dict before storing.
            any_result: Any = result
            serializable: Any = (
                any_result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            )
            await self._cache.set(key, serializable, ttl)
            return result

        return wrapper

    return decorator
