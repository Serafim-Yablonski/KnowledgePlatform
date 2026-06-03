"""Unit tests for CachedApiKeyRepository using an in-memory fake cache."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.models.api_key import ApiKey
from src.models.user import User
from src.repositories.api_key_cached import (
    CachedApiKeyRepository,
    _deserialize_api_key,
    _serialize_api_key,
)

# ---------------------------------------------------------------------------
# In-memory fake for ResponseCache
# ---------------------------------------------------------------------------

_MISS: Any = object()  # local sentinel matching cache.py behaviour


class FakeResponseCache:
    """Dict-backed stand-in for ResponseCache — no Redis required."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.delete_calls: list[str] = []

    async def get(self, key: str) -> Any:
        raw = self._store.get(key)
        if raw is None:
            return _MISS
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = json.dumps(value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self.delete_calls.append(key)

    async def get_or_set(
        self,
        key: str,
        ttl: int,
        factory: Callable[[], Awaitable[Any]],
        serializer: Callable[[Any], Any] = lambda v: v,
        deserializer: Callable[[Any], Any] = lambda v: v,
    ) -> Any:
        cached = await self.get(key)
        if cached is not _MISS:
            return deserializer(cached)
        result = await factory()
        await self.set(key, serializer(result), ttl)
        return result


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_user(user_id: uuid.UUID | None = None) -> User:
    user = User()
    user.id = user_id or uuid.uuid4()
    user.email = "test@example.com"
    user.hashed_password = "$2b$12$fakehash"
    user.is_active = True
    user.display_name = "Test User"
    user.created_at = datetime.now(UTC).replace(tzinfo=None)
    user.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return user


def _make_api_key(
    key_hash: str = "abc123",
    user: User | None = None,
    is_active: bool = True,
) -> ApiKey:
    ak = ApiKey()
    ak.id = uuid.uuid4()
    ak.user_id = uuid.uuid4()
    ak.key_hash = key_hash
    ak.prefix = "testpref"
    ak.name = "Test Key"
    ak.is_active = is_active
    ak.last_used_at = None
    ak.created_at = datetime.now(UTC).replace(tzinfo=None)
    ak.updated_at = datetime.now(UTC).replace(tzinfo=None)
    ak.user = user or _make_user(ak.user_id)  # type: ignore[assignment]
    return ak


def _make_repo(
    api_key: ApiKey | None = None,
) -> tuple[CachedApiKeyRepository, MagicMock, FakeResponseCache]:
    inner = MagicMock()
    inner.get_by_hash = AsyncMock(return_value=api_key)
    inner.list_for_user = AsyncMock(return_value=([api_key] if api_key else []))
    inner.create = AsyncMock(return_value=api_key)
    inner.count_active_for_user = AsyncMock(return_value=0)
    inner.deactivate = AsyncMock(return_value=None)

    cache = FakeResponseCache()
    repo = CachedApiKeyRepository(inner, cache)  # type: ignore[arg-type]
    return repo, inner, cache


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------


def test_api_key_serialization_round_trip() -> None:
    ak = _make_api_key()
    data = _serialize_api_key(ak)
    assert data is not None
    restored = _deserialize_api_key(data)
    assert restored is not None
    assert restored.id == ak.id
    assert restored.user_id == ak.user_id
    assert restored.key_hash == ak.key_hash
    assert restored.prefix == ak.prefix
    assert restored.name == ak.name
    assert restored.is_active == ak.is_active
    assert restored.last_used_at == ak.last_used_at
    assert restored.created_at == ak.created_at
    assert restored.updated_at == ak.updated_at


def test_serialization_restores_user_without_password() -> None:
    user = _make_user()
    ak = _make_api_key(user=user)
    data = _serialize_api_key(ak)
    assert data is not None
    # hashed_password must not appear in the serialized dict
    assert "hashed_password" not in data["user"]

    restored = _deserialize_api_key(data)
    assert restored is not None
    assert restored.user.id == user.id
    assert restored.user.email == user.email
    assert restored.user.is_active == user.is_active
    assert restored.user.display_name == user.display_name
    # placeholder must be "!" — never the original hash, never empty string
    assert restored.user.hashed_password == "!"


def test_serialize_none_returns_none() -> None:
    assert _serialize_api_key(None) is None


def test_deserialize_none_returns_none() -> None:
    assert _deserialize_api_key(None) is None


def test_serialization_preserves_last_used_at() -> None:
    ak = _make_api_key()
    ak.last_used_at = datetime(2025, 1, 15, 12, 0, 0)
    data = _serialize_api_key(ak)
    assert data is not None
    restored = _deserialize_api_key(data)
    assert restored is not None
    assert restored.last_used_at == ak.last_used_at


# ---------------------------------------------------------------------------
# get_by_hash caching
# ---------------------------------------------------------------------------


async def test_get_by_hash_cache_miss_calls_inner_and_caches() -> None:
    ak = _make_api_key(key_hash="deadbeef")
    repo, inner, cache = _make_repo(api_key=ak)

    result = await repo.get_by_hash("deadbeef")

    assert result is not None
    assert result.key_hash == "deadbeef"
    inner.get_by_hash.assert_awaited_once_with("deadbeef")
    assert "api_key:deadbeef" in cache._store


async def test_get_by_hash_cache_hit_does_not_call_inner() -> None:
    ak = _make_api_key(key_hash="deadbeef")
    repo, inner, cache = _make_repo(api_key=ak)

    await repo.get_by_hash("deadbeef")  # prime the cache
    inner.get_by_hash.reset_mock()
    result = await repo.get_by_hash("deadbeef")  # should hit cache

    assert result is not None
    assert result.key_hash == "deadbeef"
    inner.get_by_hash.assert_not_awaited()


async def test_get_by_hash_caches_none_for_unknown_key() -> None:
    repo, inner, cache = _make_repo(api_key=None)

    result = await repo.get_by_hash("unknownhash")

    assert result is None
    assert "api_key:unknownhash" in cache._store
    assert json.loads(cache._store["api_key:unknownhash"]) is None


async def test_get_by_hash_cached_none_does_not_call_inner_again() -> None:
    repo, inner, cache = _make_repo(api_key=None)

    await repo.get_by_hash("unknownhash")  # prime with None
    inner.get_by_hash.reset_mock()
    result = await repo.get_by_hash("unknownhash")  # should hit cached None

    assert result is None
    inner.get_by_hash.assert_not_awaited()


# ---------------------------------------------------------------------------
# deactivate — cache invalidation
# ---------------------------------------------------------------------------


async def test_deactivate_invalidates_cache_for_target_key() -> None:
    ak = _make_api_key(key_hash="targetkeyhash")
    repo, inner, cache = _make_repo(api_key=ak)
    inner.list_for_user = AsyncMock(return_value=[ak])

    # Prime the cache.
    await repo.get_by_hash("targetkeyhash")
    assert "api_key:targetkeyhash" in cache._store

    call_order: list[str] = []
    cache.delete_calls.clear()
    original_delete = cache.delete

    async def tracking_delete(key: str) -> None:
        call_order.append(f"delete:{key}")
        await original_delete(key)

    async def tracking_deactivate(key_id: uuid.UUID, user_id: uuid.UUID) -> None:
        call_order.append("db:deactivate")

    cache.delete = tracking_delete  # type: ignore[method-assign]
    inner.deactivate = tracking_deactivate  # type: ignore[method-assign]

    await repo.deactivate(ak.id, ak.user_id)

    # DB commit happens first, then cache invalidation (invalidate-after-commit).
    assert call_order[0] == "db:deactivate"
    assert call_order[1] == "delete:api_key:targetkeyhash"
    assert "api_key:targetkeyhash" not in cache._store


async def test_deactivate_still_calls_inner_when_key_not_in_cache() -> None:
    ak = _make_api_key(key_hash="targetkeyhash")
    repo, inner, cache = _make_repo(api_key=ak)
    inner.list_for_user = AsyncMock(return_value=[ak])

    # Do NOT prime the cache — deactivation should still proceed.
    await repo.deactivate(ak.id, ak.user_id)

    inner.deactivate.assert_awaited_once_with(ak.id, ak.user_id)


async def test_deactivate_skips_cache_delete_when_key_not_found_in_list() -> None:
    repo, inner, cache = _make_repo()
    inner.list_for_user = AsyncMock(return_value=[])  # key not owned by user

    await repo.deactivate(uuid.uuid4(), uuid.uuid4())

    assert cache.delete_calls == []
    inner.deactivate.assert_awaited_once()


# ---------------------------------------------------------------------------
# Non-cached methods delegate without touching the cache
# ---------------------------------------------------------------------------


async def test_invalidate_all_for_user_deletes_all_key_caches() -> None:
    ak1 = _make_api_key(key_hash="hash1")
    ak2 = _make_api_key(key_hash="hash2")
    repo, inner, cache = _make_repo()
    inner.list_for_user = AsyncMock(return_value=[ak1, ak2])

    # Prime the cache for both keys.
    inner.get_by_hash = AsyncMock(side_effect=[ak1, ak2])
    await repo.get_by_hash("hash1")
    await repo.get_by_hash("hash2")
    assert "api_key:hash1" in cache._store
    assert "api_key:hash2" in cache._store

    await repo.invalidate_all_for_user(ak1.user_id)

    assert "api_key:hash1" not in cache._store
    assert "api_key:hash2" not in cache._store


async def test_invalidate_all_for_user_no_keys_is_noop() -> None:
    repo, inner, cache = _make_repo()
    inner.list_for_user = AsyncMock(return_value=[])

    await repo.invalidate_all_for_user(uuid.uuid4())

    assert cache.delete_calls == []


async def test_non_cached_methods_do_not_interact_with_cache() -> None:
    ak = _make_api_key()
    repo, inner, cache = _make_repo(api_key=ak)

    await repo.create(uuid.uuid4(), "somehash", "prefix12", "My Key")
    await repo.list_for_user(uuid.uuid4())
    await repo.count_active_for_user(uuid.uuid4())

    assert cache._store == {}
    assert cache.delete_calls == []
