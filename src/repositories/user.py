import uuid

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.schemas.auth import UserCreate


class SQLAlchemyUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self._session.scalars(select(User).where(User.id == user_id))
        return result.first()

    async def get_by_email(self, email: str) -> User | None:
        # Use lower() to match the normalised email stored at registration.
        result = await self._session.scalars(
            select(User).where(User.email == email.lower())
        )
        return result.first()

    async def create(self, data: UserCreate, hashed_password: str) -> User:
        user = User(
            email=data.email,
            hashed_password=hashed_password,
            display_name=data.display_name,
        )
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def update(self, user: User, *, is_active: bool | None = None) -> User:
        if is_active is not None:
            user.is_active = is_active
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def exists_by_email(self, email: str) -> bool:
        # Check only active users — the partial unique index allows re-registration
        # after deactivation, so only active slots count as taken.
        result = await self._session.scalar(
            select(
                sa.exists().where(
                    User.email == email.lower(), User.is_active == sa.true()
                )
            )
        )
        return bool(result)
