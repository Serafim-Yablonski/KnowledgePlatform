from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import ForbiddenError
from src.models.user import User

_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with get_session() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise ForbiddenError("Missing authentication token")

    from src.repositories.user import SQLAlchemyUserRepository
    from src.services.auth import AuthService

    repo = SQLAlchemyUserRepository(session)
    service = AuthService(repo)
    return await service.get_current_user(credentials.credentials)
