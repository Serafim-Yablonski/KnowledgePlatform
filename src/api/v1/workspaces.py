import uuid

from fastapi import APIRouter, Depends, status

from src.core.dependencies import (
    get_current_user,
    get_current_workspace,
    get_workspace_service,
)
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.workspace import (
    AddMemberRequest,
    WorkspaceCreate,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)
from src.services.workspace import WorkspaceService

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    data: WorkspaceCreate,
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceResponse:
    info = await service.create(actor, data.name, data.description)
    return WorkspaceResponse(
        id=info.id,
        name=info.name,
        slug=info.slug,
        description=info.description,
        is_active=info.is_active,
        created_at=info.created_at,
        member_count=info.member_count,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> list[WorkspaceResponse]:
    infos = await service.list_for_user(actor)
    return [
        WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            description=ws.description,
            is_active=ws.is_active,
            created_at=ws.created_at,
            member_count=ws.member_count,
        )
        for ws in infos
    ]


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceResponse:
    info = await service.get_by_id(actor, workspace.id)
    return WorkspaceResponse(
        id=info.id,
        name=info.name,
        slug=info.slug,
        description=info.description,
        is_active=info.is_active,
        created_at=info.created_at,
        member_count=info.member_count,
    )


@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    data: AddMemberRequest,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceMemberResponse:
    member = await service.add_member(actor, workspace.id, data.user_email, data.role)
    return WorkspaceMemberResponse(
        user_id=member.user_id,
        email=member.email,
        display_name=member.display_name,
        role=member.role,
        joined_at=member.joined_at,
    )


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_member(
    user_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await service.remove_member(actor, workspace.id, user_id)


@router.get(
    "/{workspace_id}/members",
    response_model=list[WorkspaceMemberResponse],
)
async def list_members(
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> list[WorkspaceMemberResponse]:
    members = await service.list_members(actor, workspace.id)
    return [
        WorkspaceMemberResponse(
            user_id=m.user_id,
            email=m.email,
            display_name=m.display_name,
            role=m.role,
            joined_at=m.joined_at,
        )
        for m in members
    ]
