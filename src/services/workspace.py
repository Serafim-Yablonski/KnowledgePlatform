import uuid

import structlog
from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from src.domain.roles import ROLE_RANK, Permission, WorkspaceRole, require_permission
from src.domain.slug import generate_slug
from src.domain.workspace import WorkspaceInfo, WorkspaceMember, WorkspaceUpdateInput
from src.models.user import User
from src.models.workspace import Workspace, WorkspaceMembership
from src.repositories.protocols import (
    UserRepositoryProtocol,
    WorkspaceRepositoryProtocol,
)

logger = structlog.get_logger(__name__)


class WorkspaceService:
    def __init__(
        self,
        workspace_repo: WorkspaceRepositoryProtocol,
        user_repo: UserRepositoryProtocol,
    ) -> None:
        self._repo = workspace_repo
        self._user_repo = user_repo

    async def get_workspace_for_user(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> tuple[Workspace, WorkspaceMembership]:
        membership = await self._repo.get_membership(workspace_id, user_id)
        if membership is None:
            raise ForbiddenError("Not a member of this workspace")
        workspace = await self._repo.get_by_id(workspace_id)
        if workspace is None:
            raise ForbiddenError("Not a member of this workspace")
        return workspace, membership

    async def create(
        self,
        actor: User,
        name: str,
        description: str | None = None,
    ) -> WorkspaceInfo:
        slug = generate_slug(name)
        workspace = await self._repo.create(
            name=name,
            slug=slug,
            created_by_id=actor.id,
            description=description,
        )
        await self._repo.add_member(
            workspace_id=workspace.id,
            user_id=actor.id,
            role=WorkspaceRole.OWNER,
        )
        logger.info(
            "workspace created",
            workspace_id=str(workspace.id),
            workspace_slug=workspace.slug,
            actor_id=str(actor.id),
        )
        return WorkspaceInfo(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            description=workspace.description,
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            member_count=1,
        )

    async def get_by_id(self, workspace_id: uuid.UUID) -> WorkspaceInfo:
        workspace = await self._repo.get_by_id(workspace_id)
        if workspace is None:
            raise NotFoundError("Workspace not found")
        count = await self._repo.count_members(workspace_id)
        return WorkspaceInfo(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            description=workspace.description,
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            member_count=count,
        )

    async def update(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        role: WorkspaceRole,
        data: WorkspaceUpdateInput,
    ) -> WorkspaceInfo:
        require_permission(role, Permission.UPDATE_WORKSPACE)
        workspace = await self._repo.update(workspace_id, data)
        count = await self._repo.count_members(workspace_id)
        return WorkspaceInfo(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            description=workspace.description,
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            member_count=count,
        )

    async def list_for_user(self, actor: User) -> list[WorkspaceInfo]:
        rows = await self._repo.list_for_user_with_counts(actor.id)
        return [
            WorkspaceInfo(
                id=ws.id,
                name=ws.name,
                slug=ws.slug,
                description=ws.description,
                is_active=ws.is_active,
                created_at=ws.created_at,
                member_count=count,
            )
            for ws, count in rows
        ]

    async def add_member(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        actor_role: WorkspaceRole,
        user_email: str,
        role: WorkspaceRole = WorkspaceRole.MEMBER,
    ) -> WorkspaceMember:
        require_permission(actor_role, Permission.MANAGE_MEMBERS)

        # Prevent privilege escalation: a role can only be granted if the actor
        # holds an equal or higher rank (e.g. ADMIN cannot make someone OWNER).
        if ROLE_RANK[role] > ROLE_RANK[actor_role]:
            raise ForbiddenError("Cannot grant a role higher than your own")

        target = await self._user_repo.get_by_email(user_email)
        # Return the same error regardless of whether the email is unknown or
        # already a member — prevents leaking which emails are registered to
        # ADMIN-level callers who have manage_members permission.
        if target is None:
            raise ConflictError("Invitation could not be sent")
        existing = await self._repo.get_membership(workspace_id, target.id)
        if existing is not None:
            raise ConflictError("Invitation could not be sent")

        try:
            new_membership = await self._repo.add_member(
                workspace_id=workspace_id,
                user_id=target.id,
                role=role,
                invited_by_id=actor.id,
            )
        except IntegrityError as exc:
            raise ConflictError("Invitation could not be sent") from exc

        logger.info(
            "workspace member added",
            workspace_id=str(workspace_id),
            actor_id=str(actor.id),
            target_user_id=str(target.id),
            role=role.value,
        )
        return WorkspaceMember(
            user_id=target.id,
            email=target.email,
            display_name=target.display_name,
            role=new_membership.role,
            joined_at=new_membership.joined_at,
        )

    async def remove_member(
        self,
        workspace_id: uuid.UUID,
        actor_role: WorkspaceRole,
        target_user_id: uuid.UUID,
    ) -> None:
        require_permission(actor_role, Permission.MANAGE_MEMBERS)

        target_membership = await self._repo.get_membership(
            workspace_id, target_user_id
        )
        if target_membership is None:
            raise NotFoundError("User is not a member of this workspace")

        # Prevent privilege escalation: a role can only remove members of equal
        # or lower rank (e.g. ADMIN cannot remove an OWNER).
        if ROLE_RANK[target_membership.role] > ROLE_RANK[actor_role]:
            raise ForbiddenError("Cannot remove a member with a higher role")

        if target_membership.role == WorkspaceRole.OWNER:
            owner_count = await self._repo.count_owners_for_update(workspace_id)
            if owner_count <= 1:
                raise ConflictError("Cannot remove the last owner of a workspace")

        await self._repo.remove_member(workspace_id, target_user_id)
        logger.info(
            "workspace member removed",
            workspace_id=str(workspace_id),
            actor_role=actor_role.value,
            target_user_id=str(target_user_id),
        )

    async def delete(
        self,
        actor: User,
        workspace: Workspace,
        role: WorkspaceRole,
    ) -> None:
        require_permission(role, Permission.DELETE_WORKSPACE)
        await self._repo.delete(workspace.id)
        logger.info(
            "workspace deleted",
            workspace_id=str(workspace.id),
            actor_id=str(actor.id),
        )

    async def get_user_role(
        self, actor: User, workspace_id: uuid.UUID
    ) -> WorkspaceRole:
        _, membership = await self.get_workspace_for_user(workspace_id, actor.id)
        return membership.role

    async def list_members(self, workspace_id: uuid.UUID) -> list[WorkspaceMember]:
        members = await self._repo.list_members(workspace_id)
        return [
            WorkspaceMember(
                user_id=m.user_id,
                email=m.user.email,
                display_name=m.user.display_name,
                role=m.role,
                joined_at=m.joined_at,
            )
            for m in members
        ]
