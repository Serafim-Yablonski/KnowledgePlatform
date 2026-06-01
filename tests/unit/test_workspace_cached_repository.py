"""Unit tests for CachedWorkspaceRepository using an in-memory fake cache."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.domain.roles import WorkspaceRole
from src.domain.workspace import WorkspaceUpdateInput
from src.models.workspace import Workspace, WorkspaceMembership
from src.repositories.workspace_cached import (
    CachedWorkspaceRepository,
    _deserialize_membership,
    _deserialize_workspace,
    _serialize_membership,
    _serialize_workspace,
)

# ---------------------------------------------------------------------------
# In-memory fake for ResponseCache
# ---------------------------------------------------------------------------

_MISS: Any = object()  # local sentinel matching cache.py behaviour


class FakeResponseCache:
    """Dict-backed stand-in for ResponseCache — no Redis required."""

    def __init__(self) -> None:
        # Store serialised JSON strings exactly as Redis would.
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

    async def delete_pattern(self, pattern: str) -> int:
        import fnmatch

        to_delete = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for k in to_delete:
            del self._store[k]
        return len(to_delete)

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


def _make_workspace() -> Workspace:
    ws = Workspace()
    ws.id = uuid.uuid4()
    ws.name = "Test WS"
    ws.slug = "test-ws-abcd"
    ws.description = "A description"
    ws.created_by = uuid.uuid4()
    ws.is_active = True
    ws.created_at = datetime.now(UTC).replace(tzinfo=None)
    ws.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return ws


def _make_membership(
    workspace_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    role: WorkspaceRole = WorkspaceRole.MEMBER,
) -> WorkspaceMembership:
    m = WorkspaceMembership()
    m.workspace_id = workspace_id or uuid.uuid4()
    m.user_id = user_id or uuid.uuid4()
    m.role = role
    m.invited_by = None
    m.joined_at = datetime.now(UTC).replace(tzinfo=None)
    return m


def _make_repo(
    workspace: Workspace | None = None,
    membership: WorkspaceMembership | None = None,
) -> tuple[CachedWorkspaceRepository, MagicMock, FakeResponseCache]:
    inner = MagicMock()
    inner.get_by_id = AsyncMock(return_value=workspace)
    inner.get_membership = AsyncMock(return_value=membership)
    inner.update = AsyncMock(return_value=workspace or _make_workspace())
    inner.add_member = AsyncMock(return_value=membership)
    inner.remove_member = AsyncMock(return_value=None)
    inner.create = AsyncMock(return_value=workspace)
    inner.get_by_slug = AsyncMock(return_value=workspace)
    inner.list_for_user = AsyncMock(return_value=[])
    inner.list_members = AsyncMock(return_value=[])
    inner.count_members = AsyncMock(return_value=0)
    inner.count_owners_for_update = AsyncMock(return_value=0)
    inner.delete = AsyncMock(return_value=None)

    cache = FakeResponseCache()
    repo = CachedWorkspaceRepository(inner, cache)  # type: ignore[arg-type]
    return repo, inner, cache


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------


def test_workspace_serialization_round_trip() -> None:
    ws = _make_workspace()
    data = _serialize_workspace(ws)
    assert data is not None
    restored = _deserialize_workspace(data)
    assert restored is not None
    assert restored.id == ws.id
    assert restored.name == ws.name
    assert restored.slug == ws.slug
    assert restored.description == ws.description
    assert restored.created_by == ws.created_by
    assert restored.is_active == ws.is_active
    assert restored.created_at == ws.created_at
    assert restored.updated_at == ws.updated_at


def test_membership_serialization_round_trip() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    m = _make_membership(workspace_id=ws_id, user_id=user_id, role=WorkspaceRole.ADMIN)
    data = _serialize_membership(m)
    assert data is not None
    restored = _deserialize_membership(data)
    assert restored is not None
    assert restored.workspace_id == ws_id
    assert restored.user_id == user_id
    assert restored.role == WorkspaceRole.ADMIN
    assert restored.invited_by is None
    assert restored.joined_at == m.joined_at


def test_serialize_none_returns_none() -> None:
    assert _serialize_workspace(None) is None
    assert _serialize_membership(None) is None


def test_deserialize_none_returns_none() -> None:
    assert _deserialize_workspace(None) is None
    assert _deserialize_membership(None) is None


# ---------------------------------------------------------------------------
# get_by_id caching
# ---------------------------------------------------------------------------


async def test_get_by_id_cache_miss_calls_inner_and_caches() -> None:
    ws = _make_workspace()
    repo, inner, cache = _make_repo(workspace=ws)

    result = await repo.get_by_id(ws.id)

    assert result is not None
    assert result.id == ws.id
    inner.get_by_id.assert_awaited_once_with(ws.id)
    assert f"workspace:{ws.id}" in cache._store


async def test_get_by_id_cache_hit_does_not_call_inner() -> None:
    ws = _make_workspace()
    repo, inner, cache = _make_repo(workspace=ws)

    await repo.get_by_id(ws.id)  # prime the cache
    inner.get_by_id.reset_mock()
    result = await repo.get_by_id(ws.id)  # should hit cache

    assert result is not None
    assert result.id == ws.id
    inner.get_by_id.assert_not_awaited()


async def test_get_by_id_caches_none_result() -> None:
    repo, inner, cache = _make_repo(workspace=None)
    ws_id = uuid.uuid4()

    result = await repo.get_by_id(ws_id)

    assert result is None
    assert f"workspace:{ws_id}" in cache._store
    # Cached value is JSON null.
    assert json.loads(cache._store[f"workspace:{ws_id}"]) is None


async def test_get_by_id_cached_none_does_not_call_inner_again() -> None:
    repo, inner, cache = _make_repo(workspace=None)
    ws_id = uuid.uuid4()

    await repo.get_by_id(ws_id)  # prime with None
    inner.get_by_id.reset_mock()
    result = await repo.get_by_id(ws_id)  # should hit cached None

    assert result is None
    inner.get_by_id.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_membership caching
# ---------------------------------------------------------------------------


async def test_get_membership_cache_miss_calls_inner_and_caches() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    m = _make_membership(workspace_id=ws_id, user_id=user_id, role=WorkspaceRole.OWNER)
    repo, inner, cache = _make_repo(membership=m)

    result = await repo.get_membership(ws_id, user_id)

    assert result is not None
    assert result.role == WorkspaceRole.OWNER
    inner.get_membership.assert_awaited_once_with(ws_id, user_id)
    assert f"membership:{ws_id}:{user_id}" in cache._store


async def test_get_membership_cache_hit_does_not_call_inner() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    m = _make_membership(workspace_id=ws_id, user_id=user_id)
    repo, inner, cache = _make_repo(membership=m)

    await repo.get_membership(ws_id, user_id)
    inner.get_membership.reset_mock()
    result = await repo.get_membership(ws_id, user_id)

    assert result is not None
    inner.get_membership.assert_not_awaited()


async def test_get_membership_caches_none_for_non_member() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    repo, inner, cache = _make_repo(membership=None)

    result = await repo.get_membership(ws_id, user_id)

    assert result is None
    assert f"membership:{ws_id}:{user_id}" in cache._store


# ---------------------------------------------------------------------------
# Mutation invalidation
# ---------------------------------------------------------------------------


async def test_update_invalidates_workspace_cache_before_db_write() -> None:
    ws = _make_workspace()
    repo, inner, cache = _make_repo(workspace=ws)

    # Prime the cache.
    await repo.get_by_id(ws.id)
    assert f"workspace:{ws.id}" in cache._store

    call_order: list[str] = []
    cache.delete_calls.clear()

    original_delete = cache.delete

    async def tracking_delete(key: str) -> None:
        call_order.append(f"delete:{key}")
        await original_delete(key)

    async def tracking_inner_update(
        wid: uuid.UUID, d: WorkspaceUpdateInput
    ) -> Workspace:
        call_order.append("db:update")
        return ws

    cache.delete = tracking_delete  # type: ignore[method-assign]
    inner.update = tracking_inner_update  # type: ignore[method-assign]

    await repo.update(ws.id, WorkspaceUpdateInput(name="New Name"))

    assert call_order[0] == f"delete:workspace:{ws.id}"
    assert call_order[1] == "db:update"
    assert f"workspace:{ws.id}" not in cache._store


async def test_add_member_invalidates_membership_cache_before_db_write() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    m = _make_membership(workspace_id=ws_id, user_id=user_id)
    repo, inner, cache = _make_repo(membership=m)

    # Prime with None (user not yet a member).
    inner.get_membership = AsyncMock(return_value=None)
    await repo.get_membership(ws_id, user_id)
    assert f"membership:{ws_id}:{user_id}" in cache._store

    call_order: list[str] = []
    original_delete = cache.delete

    async def tracking_delete(key: str) -> None:
        call_order.append(f"delete:{key}")
        await original_delete(key)

    async def tracking_inner_add_member(
        *args: Any, **kwargs: Any
    ) -> WorkspaceMembership:
        call_order.append("db:add_member")
        return m

    cache.delete = tracking_delete  # type: ignore[method-assign]
    inner.add_member = tracking_inner_add_member  # type: ignore[method-assign]

    await repo.add_member(ws_id, user_id, WorkspaceRole.MEMBER)

    assert call_order[0] == f"delete:membership:{ws_id}:{user_id}"
    assert call_order[1] == "db:add_member"
    assert f"membership:{ws_id}:{user_id}" not in cache._store


async def test_remove_member_invalidates_membership_cache_before_db_write() -> None:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    m = _make_membership(workspace_id=ws_id, user_id=user_id)
    repo, inner, cache = _make_repo(membership=m)

    await repo.get_membership(ws_id, user_id)  # prime
    assert f"membership:{ws_id}:{user_id}" in cache._store

    call_order: list[str] = []
    original_delete = cache.delete

    async def tracking_delete(key: str) -> None:
        call_order.append(f"delete:{key}")
        await original_delete(key)

    async def tracking_inner_remove(*args: Any) -> None:
        call_order.append("db:remove_member")

    cache.delete = tracking_delete  # type: ignore[method-assign]
    inner.remove_member = tracking_inner_remove  # type: ignore[method-assign]

    await repo.remove_member(ws_id, user_id)

    assert call_order[0] == f"delete:membership:{ws_id}:{user_id}"
    assert call_order[1] == "db:remove_member"
    assert f"membership:{ws_id}:{user_id}" not in cache._store


# ---------------------------------------------------------------------------
# delete — invalidates workspace key and all membership keys for the workspace
# ---------------------------------------------------------------------------


async def test_delete_invalidates_workspace_and_all_membership_keys() -> None:
    ws = _make_workspace()
    user_id_1 = uuid.uuid4()
    user_id_2 = uuid.uuid4()
    m1 = _make_membership(workspace_id=ws.id, user_id=user_id_1)
    m2 = _make_membership(workspace_id=ws.id, user_id=user_id_2)
    repo, inner, cache = _make_repo(workspace=ws, membership=m1)
    inner.get_membership = AsyncMock(side_effect=[m1, m2])

    # Prime workspace and two membership cache entries.
    await repo.get_by_id(ws.id)
    await repo.get_membership(ws.id, user_id_1)
    await repo.get_membership(ws.id, user_id_2)
    assert f"workspace:{ws.id}" in cache._store
    assert f"membership:{ws.id}:{user_id_1}" in cache._store
    assert f"membership:{ws.id}:{user_id_2}" in cache._store

    await repo.delete(ws.id)

    assert f"workspace:{ws.id}" not in cache._store
    assert f"membership:{ws.id}:{user_id_1}" not in cache._store
    assert f"membership:{ws.id}:{user_id_2}" not in cache._store
    inner.delete.assert_awaited_once_with(ws.id)


# ---------------------------------------------------------------------------
# Non-cached methods delegate to inner without touching cache
# ---------------------------------------------------------------------------


async def test_non_cached_methods_do_not_interact_with_cache() -> None:
    ws = _make_workspace()
    repo, inner, cache = _make_repo(workspace=ws)

    await repo.create("name", "slug", uuid.uuid4(), "desc")
    await repo.get_by_slug("slug")
    await repo.list_for_user(uuid.uuid4())
    await repo.list_members(ws.id)
    await repo.count_members(ws.id)
    await repo.count_owners_for_update(ws.id)

    # None of these calls should have written anything to the cache store.
    assert cache._store == {}
    assert cache.delete_calls == []
