from __future__ import annotations

import hashlib
import secrets
import uuid

import structlog
from sqlalchemy.exc import SQLAlchemyError

from src.core.cache import CacheKeys, ResponseCache
from src.core.config import settings
from src.core.exceptions import ConflictError, InvalidTokenError, NotFoundError
from src.models.api_key import ApiKey
from src.models.user import User
from src.repositories.protocols import UnitOfWorkFactory

logger = structlog.get_logger(__name__)

_MAX_ACTIVE_KEYS = 5


class ApiKeyService:
    def __init__(self, uow: UnitOfWorkFactory, claims: ResponseCache) -> None:
        self._uow = uow
        self._claims = claims

    async def create(self, user_id: uuid.UUID, name: str) -> tuple[ApiKey, str]:
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        async with self._uow() as uow:
            active = await uow.api_keys.count_active_for_user(user_id)
            if active >= _MAX_ACTIVE_KEYS:
                raise ConflictError(
                    f"Maximum of {_MAX_ACTIVE_KEYS} active API keys reached. "
                    "Deactivate an existing key before creating a new one."
                )
            api_key = await uow.api_keys.create(
                user_id=user_id, key_hash=key_hash, prefix=raw_key[:8], name=name
            )
            await uow.commit()
        return api_key, raw_key

    async def authenticate(self, raw_key: str) -> User:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        async with self._uow() as uow:
            api_key = await uow.api_keys.get_by_hash(key_hash)
        if api_key is None or not api_key.is_active:
            raise InvalidTokenError("Invalid or revoked API key")
        if not api_key.user.is_active:
            raise InvalidTokenError("Invalid or revoked API key")
        # After the read's block: the write gets its own unit of work, so its
        # failure can't touch what authentication returns (ADR-019, ADR-033).
        await self._record_use(api_key.id)
        return api_key.user

    async def _record_use(self, key_id: uuid.UUID) -> None:
        # Throttled (ADR-019): the first request in each window writes, the rest
        # skip the DB. Without Redis the claim fails closed and nothing is written.
        if not await self._claims.claim(
            CacheKeys.api_key_touched(key_id), settings.API_KEY_LAST_USED_RESOLUTION
        ):
            return
        # Bookkeeping must never turn a valid key into a failed request. Only the
        # database's errors (a refused connection is an OSError): anything else, a
        # unit of work opened inside another, is a bug, not a failed write.
        try:
            async with self._uow() as uow:
                await uow.api_keys.touch_last_used(key_id)
                await uow.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(
                "failed to record api key use", key_id=str(key_id), error=repr(exc)
            )

    async def list_for_user(self, user_id: uuid.UUID) -> list[ApiKey]:
        async with self._uow() as uow:
            return await uow.api_keys.list_for_user(user_id)

    async def deactivate(
        self, key_id: uuid.UUID, requesting_user_id: uuid.UUID
    ) -> None:
        async with self._uow() as uow:
            keys = await uow.api_keys.list_for_user(requesting_user_id)
            if not any(k.id == key_id for k in keys):
                raise NotFoundError("API key not found")
            await uow.api_keys.deactivate(key_id, requesting_user_id)
            await uow.commit()
