import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import ForbiddenError
from src.models.user import User
from src.models.workspace import Workspace

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


async def get_current_workspace(
    workspace_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> Workspace:
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository

    repo = SQLAlchemyWorkspaceRepository(session)
    membership = await repo.get_membership(workspace_id, user.id)
    if membership is None:
        raise ForbiddenError("Not a member of this workspace")
    request.state.workspace_role = membership.role
    workspace = await repo.get_by_id(workspace_id)
    if workspace is None:
        raise ForbiddenError("Not a member of this workspace")
    return workspace
