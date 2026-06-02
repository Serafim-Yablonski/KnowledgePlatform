from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from src.core.cache import CacheKeys, ResponseCache
from src.core.config import settings
from src.domain.roles import WorkspaceRole
from src.domain.workspace import WorkspaceUpdateInput
from src.models.workspace import Workspace, WorkspaceMembership
from src.repositories.protocols import WorkspaceRepositoryProtocol

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

# Both models have lazy="raise" on all relationships, so only column values are
# accessed here. Deserialized instances are plain Python objects with no session
# attached — safe because callers of get_by_id and get_membership only access
# column attributes, never relationship traversals.


def _serialize_workspace(ws: Workspace | None) -> dict[str, Any] | None:
    if ws is None:
        return None
    return {
        "id": str(ws.id),
        "name": ws.name,
        "slug": ws.slug,
        "description": ws.description,
        "created_by": str(ws.created_by),
        "is_active": ws.is_active,
        "created_at": ws.created_at.isoformat(),
        "updated_at": ws.updated_at.isoformat(),
    }


def _deserialize_workspace(data: Any) -> Workspace | None:
    if data is None:
        return None
    ws = Workspace()
    ws.id = uuid.UUID(data["id"])
    ws.name = data["name"]
    ws.slug = data["slug"]
    ws.description = data["description"]
    ws.created_by = uuid.UUID(data["created_by"])
    ws.is_active = data["is_active"]
    ws.created_at = datetime.fromisoformat(data["created_at"])
    ws.updated_at = datetime.fromisoformat(data["updated_at"])
    return ws


def _serialize_membership(m: WorkspaceMembership | None) -> dict[str, Any] | None:
    if m is None:
        return None
    return {
        "workspace_id": str(m.workspace_id),
        "user_id": str(m.user_id),
        "role": m.role,
        "invited_by": str(m.invited_by) if m.invited_by else None,
        "joined_at": m.joined_at.isoformat(),
    }


def _deserialize_membership(data: Any) -> WorkspaceMembership | None:
    if data is None:
        return None
    m = WorkspaceMembership()
    m.workspace_id = uuid.UUID(data["workspace_id"])
    m.user_id = uuid.UUID(data["user_id"])
    m.role = WorkspaceRole(data["role"])
    m.invited_by = uuid.UUID(data["invited_by"]) if data["invited_by"] else None
    m.joined_at = datetime.fromisoformat(data["joined_at"])
    return m


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


class CachedWorkspaceRepository:
    """Wraps WorkspaceRepositoryProtocol with a read-through cache for
    get_by_id and get_membership. Mutations commit via the inner repository
    first, then invalidate the cache — this prevents a concurrent reader from
    re-populating the cache with pre-commit (stale) data during the write
    window."""

    def __init__(
        self,
        inner: WorkspaceRepositoryProtocol,
        cache: ResponseCache,
    ) -> None:
        self._inner = inner
        self._cache = cache

    # ------------------------------------------------------------------
    # Cached reads
    # ------------------------------------------------------------------

    async def get_by_id(self, workspace_id: uuid.UUID) -> Workspace | None:
        return await self._cache.get_or_set(
            key=CacheKeys.workspace(workspace_id),
            ttl=settings.CACHE_TTL_WORKSPACE,
            factory=lambda: self._inner.get_by_id(workspace_id),
            serializer=_serialize_workspace,
            deserializer=_deserialize_workspace,
        )

    async def get_membership(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMembership | None:
        return await self._cache.get_or_set(
            key=CacheKeys.membership(workspace_id, user_id),
            ttl=settings.CACHE_TTL_MEMBERSHIP,
            factory=lambda: self._inner.get_membership(workspace_id, user_id),
            serializer=_serialize_membership,
            deserializer=_deserialize_membership,
        )

    # ------------------------------------------------------------------
    # Mutations — commit via inner repo first, then invalidate cache.
    # ------------------------------------------------------------------

    async def update(
        self, workspace_id: uuid.UUID, data: WorkspaceUpdateInput
    ) -> Workspace:
        result = await self._inner.update(workspace_id, data)
        await self._cache.delete(CacheKeys.workspace(workspace_id))
        return result

    async def add_member(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: WorkspaceRole,
        invited_by_id: uuid.UUID | None = None,
    ) -> WorkspaceMembership:
        result = await self._inner.add_member(
            workspace_id, user_id, role, invited_by_id
        )
        await self._cache.delete(CacheKeys.membership(workspace_id, user_id))
        return result

    async def remove_member(self, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
        await self._inner.remove_member(workspace_id, user_id)
        await self._cache.delete(CacheKeys.membership(workspace_id, user_id))

    async def delete(self, workspace_id: uuid.UUID) -> None:
        await self._inner.delete(workspace_id)
        await self._cache.delete(CacheKeys.workspace(workspace_id))
        await self._cache.delete_pattern(CacheKeys.membership_pattern(workspace_id))

    # ------------------------------------------------------------------
    # Pure delegation — no caching
    # ------------------------------------------------------------------

    async def create(
        self,
        name: str,
        slug: str,
        created_by_id: uuid.UUID,
        description: str | None = None,
    ) -> Workspace:
        return await self._inner.create(name, slug, created_by_id, description)

    async def get_by_slug(self, slug: str) -> Workspace | None:
        return await self._inner.get_by_slug(slug)

    async def list_for_user(self, user_id: uuid.UUID) -> list[Workspace]:
        return await self._inner.list_for_user(user_id)

    async def list_for_user_with_counts(
        self, user_id: uuid.UUID
    ) -> list[tuple[Workspace, int]]:
        return await self._inner.list_for_user_with_counts(user_id)

    async def list_members(self, workspace_id: uuid.UUID) -> list[WorkspaceMembership]:
        return await self._inner.list_members(workspace_id)

    async def count_members(self, workspace_id: uuid.UUID) -> int:
        return await self._inner.count_members(workspace_id)

    async def count_owners_for_update(self, workspace_id: uuid.UUID) -> int:
        return await self._inner.count_owners_for_update(workspace_id)
