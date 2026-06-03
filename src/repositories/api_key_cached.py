from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from src.core.cache import CacheKeys, ResponseCache
from src.core.config import settings
from src.models.api_key import ApiKey
from src.models.user import User
from src.repositories.protocols import ApiKeyRepositoryProtocol

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

# Both models have lazy="raise" on relationships so only column values are
# accessed here. Deserialized instances are detached Python objects with no
# session attached — callers only access column attributes, never relationships.
# hashed_password is intentionally excluded from serialization: route handlers
# never need it after auth, and there is no reason to store credential-adjacent
# data in Redis. The placeholder "!" (Unix convention for a locked account)
# is set on deserialization so the attribute is structurally valid and will
# never pass bcrypt verification.


def _serialize_api_key(api_key: ApiKey | None) -> dict[str, Any] | None:
    if api_key is None:
        return None
    user = api_key.user
    return {
        "id": str(api_key.id),
        "user_id": str(api_key.user_id),
        "key_hash": api_key.key_hash,
        "prefix": api_key.prefix,
        "name": api_key.name,
        "is_active": api_key.is_active,
        "last_used_at": (
            api_key.last_used_at.isoformat() if api_key.last_used_at else None
        ),
        "created_at": api_key.created_at.isoformat(),
        "updated_at": api_key.updated_at.isoformat(),
        "user": {
            "id": str(user.id),
            "email": user.email,
            "is_active": user.is_active,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        },
    }


def _deserialize_api_key(data: dict[str, Any] | None) -> ApiKey | None:
    if data is None:
        return None

    user = User()
    user_data = data["user"]
    user.id = uuid.UUID(user_data["id"])
    user.email = user_data["email"]
    user.is_active = user_data["is_active"]
    user.display_name = user_data["display_name"]
    user.hashed_password = "!"
    user.created_at = datetime.fromisoformat(user_data["created_at"])
    user.updated_at = datetime.fromisoformat(user_data["updated_at"])

    api_key = ApiKey()
    api_key.id = uuid.UUID(data["id"])
    api_key.user_id = uuid.UUID(data["user_id"])
    api_key.key_hash = data["key_hash"]
    api_key.prefix = data["prefix"]
    api_key.name = data["name"]
    api_key.is_active = data["is_active"]
    api_key.last_used_at = (
        datetime.fromisoformat(data["last_used_at"]) if data["last_used_at"] else None
    )
    api_key.created_at = datetime.fromisoformat(data["created_at"])
    api_key.updated_at = datetime.fromisoformat(data["updated_at"])
    api_key.user = user
    return api_key


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


class CachedApiKeyRepository:
    """Wraps ApiKeyRepositoryProtocol with a read-through cache for get_by_hash.

    deactivate() commits the DB change first, then removes the cache entry.
    This prevents a concurrent auth request from re-populating the cache with
    an active key during the window between cache delete and DB commit.
    """

    def __init__(
        self,
        inner: ApiKeyRepositoryProtocol,
        cache: ResponseCache,
    ) -> None:
        self._inner = inner
        self._cache = cache

    # ------------------------------------------------------------------
    # Cached reads
    # ------------------------------------------------------------------

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        return await self._cache.get_or_set(
            key=CacheKeys.api_key(key_hash),
            ttl=settings.CACHE_TTL_API_KEY,
            factory=lambda: self._inner.get_by_hash(key_hash),
            serializer=_serialize_api_key,
            deserializer=_deserialize_api_key,
        )

    # ------------------------------------------------------------------
    # Mutations — commit via inner repo first, then invalidate cache.
    # ------------------------------------------------------------------

    async def deactivate(self, key_id: uuid.UUID, user_id: uuid.UUID) -> None:
        # Fetch the hash before the DB write so we still have it after deactivation.
        # deactivate() is a rare admin operation so the extra DB call is acceptable.
        keys = await self._inner.list_for_user(user_id)
        target = next((k for k in keys if k.id == key_id), None)
        await self._inner.deactivate(key_id, user_id)
        if target is not None:
            await self._cache.delete(CacheKeys.api_key(target.key_hash))

    async def invalidate_all_for_user(self, user_id: uuid.UUID) -> None:
        keys = await self._inner.list_for_user(user_id)
        for key in keys:
            await self._cache.delete(CacheKeys.api_key(key.key_hash))

    # ------------------------------------------------------------------
    # Pure delegation — no caching
    # ------------------------------------------------------------------

    async def create(
        self,
        user_id: uuid.UUID,
        key_hash: str,
        prefix: str,
        name: str,
    ) -> ApiKey:
        return await self._inner.create(user_id, key_hash, prefix, name)

    async def list_for_user(self, user_id: uuid.UUID) -> list[ApiKey]:
        return await self._inner.list_for_user(user_id)

    async def count_active_for_user(self, user_id: uuid.UUID) -> int:
        return await self._inner.count_active_for_user(user_id)
