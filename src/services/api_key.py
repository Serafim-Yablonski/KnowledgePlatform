from __future__ import annotations

import hashlib
import secrets
import uuid

from src.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from src.models.api_key import ApiKey
from src.models.user import User
from src.repositories.protocols import ApiKeyRepositoryProtocol

_MAX_ACTIVE_KEYS = 5


class ApiKeyService:
    def __init__(self, repo: ApiKeyRepositoryProtocol) -> None:
        self._repo = repo

    async def create(self, user_id: uuid.UUID, name: str) -> tuple[ApiKey, str]:
        active = await self._repo.count_active_for_user(user_id)
        if active >= _MAX_ACTIVE_KEYS:
            raise ConflictError(
                f"Maximum of {_MAX_ACTIVE_KEYS} active API keys reached. "
                "Deactivate an existing key before creating a new one."
            )
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        prefix = raw_key[:8]
        api_key = await self._repo.create(
            user_id=user_id, key_hash=key_hash, prefix=prefix, name=name
        )
        return api_key, raw_key

    async def authenticate(self, raw_key: str) -> User:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = await self._repo.get_by_hash(key_hash)
        if api_key is None or not api_key.is_active:
            raise ForbiddenError("Invalid or revoked API key")
        return api_key.user

    async def list_for_user(self, user_id: uuid.UUID) -> list[ApiKey]:
        return await self._repo.list_for_user(user_id)

    async def deactivate(
        self, key_id: uuid.UUID, requesting_user_id: uuid.UUID
    ) -> None:
        keys = await self._repo.list_for_user(requesting_user_id)
        target = next((k for k in keys if k.id == key_id), None)
        if target is None:
            raise NotFoundError("API key not found")
        await self._repo.deactivate(key_id, requesting_user_id)
