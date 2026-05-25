from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.api_key import ApiKey


class SQLAlchemyApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: uuid.UUID,
        key_hash: str,
        prefix: str,
        name: str,
    ) -> ApiKey:
        key = ApiKey(user_id=user_id, key_hash=key_hash, prefix=prefix, name=name)
        self._session.add(key)
        await self._session.commit()
        await self._session.refresh(key)
        return key

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        result = await self._session.scalars(
            select(ApiKey)
            .where(ApiKey.key_hash == key_hash)
            .options(selectinload(ApiKey.user))
        )
        return result.first()

    async def list_for_user(self, user_id: uuid.UUID) -> list[ApiKey]:
        result = await self._session.scalars(
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_at.desc())
        )
        return list(result.all())

    async def count_active_for_user(self, user_id: uuid.UUID) -> int:
        result = await self._session.scalar(
            select(func.count())
            .select_from(ApiKey)
            .where(ApiKey.user_id == user_id, ApiKey.is_active == sa.true())
        )
        return int(result or 0)

    async def deactivate(self, key_id: uuid.UUID, user_id: uuid.UUID) -> None:
        await self._session.execute(
            sa.update(ApiKey)
            .where(ApiKey.id == key_id, ApiKey.user_id == user_id)
            .values(is_active=False)
        )
        await self._session.commit()
