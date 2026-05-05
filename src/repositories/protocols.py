import uuid
from typing import Protocol

from src.models.user import User
from src.schemas.auth import UserCreate


class UserRepositoryProtocol(Protocol):
    async def get_by_id(self, user_id: uuid.UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def create(self, data: UserCreate, hashed_password: str) -> User: ...

    async def update(self, user: User, *, is_active: bool | None = None) -> User: ...

    async def exists_by_email(self, email: str) -> bool: ...
