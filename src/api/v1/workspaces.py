import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.dependencies import get_current_user, get_current_workspace, get_db
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.workspace import (
    AddMemberRequest,
    WorkspaceCreate,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)
from src.services.workspace import WorkspaceService

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])


def _workspace_service(session: AsyncSession = Depends(get_db)) -> WorkspaceService:
    return WorkspaceService(
        workspace_repo=SQLAlchemyWorkspaceRepository(session),
        user_repo=SQLAlchemyUserRepository(session),
    )


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    data: WorkspaceCreate,
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> WorkspaceResponse:
    return await service.create(actor, data)


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> list[WorkspaceResponse]:
    return await service.list_for_user(actor)


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> WorkspaceResponse:
    return await service.get_by_id(actor, workspace.id)


@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    data: AddMemberRequest,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> WorkspaceMemberResponse:
    return await service.add_member(actor, workspace.id, data)


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_member(
    user_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> None:
    await service.remove_member(actor, workspace.id, user_id)


@router.get(
    "/{workspace_id}/members",
    response_model=list[WorkspaceMemberResponse],
)
async def list_members(
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_workspace_service),
) -> list[WorkspaceMemberResponse]:
    return await service.list_members(actor, workspace.id)
