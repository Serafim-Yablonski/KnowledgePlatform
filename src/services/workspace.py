import uuid

from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from src.domain.roles import PERMISSIONS, ROLE_RANK, WorkspaceRole
from src.domain.slug import generate_slug
from src.domain.workspace import WorkspaceInfo, WorkspaceMember
from src.models.user import User
from src.models.workspace import Workspace, WorkspaceMembership
from src.repositories.protocols import (
    UserRepositoryProtocol,
    WorkspaceRepositoryProtocol,
)


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
        return WorkspaceInfo(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            description=workspace.description,
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            member_count=1,
        )

    async def get_by_id(self, actor: User, workspace_id: uuid.UUID) -> WorkspaceInfo:
        membership = await self._repo.get_membership(workspace_id, actor.id)
        if membership is None:
            raise ForbiddenError("Not a member of this workspace")
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

    async def list_for_user(self, actor: User) -> list[WorkspaceInfo]:
        workspaces = await self._repo.list_for_user(actor.id)
        result: list[WorkspaceInfo] = []
        for ws in workspaces:
            count = await self._repo.count_members(ws.id)
            result.append(
                WorkspaceInfo(
                    id=ws.id,
                    name=ws.name,
                    slug=ws.slug,
                    description=ws.description,
                    is_active=ws.is_active,
                    created_at=ws.created_at,
                    member_count=count,
                )
            )
        return result

    async def add_member(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        user_email: str,
        role: WorkspaceRole = WorkspaceRole.MEMBER,
    ) -> WorkspaceMember:
        membership = await self._repo.get_membership(workspace_id, actor.id)
        if membership is None or "manage_members" not in PERMISSIONS[membership.role]:
            raise ForbiddenError("Insufficient permissions to add members")

        # Prevent privilege escalation: a role can only be granted if the actor
        # holds an equal or higher rank (e.g. ADMIN cannot make someone OWNER).
        if ROLE_RANK[role] > ROLE_RANK[membership.role]:
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

        return WorkspaceMember(
            user_id=target.id,
            email=target.email,
            display_name=target.display_name,
            role=new_membership.role,
            joined_at=new_membership.joined_at,
        )

    async def remove_member(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        target_user_id: uuid.UUID,
    ) -> None:
        actor_membership = await self._repo.get_membership(workspace_id, actor.id)
        if (
            actor_membership is None
            or "manage_members" not in PERMISSIONS[actor_membership.role]
        ):
            raise ForbiddenError("Insufficient permissions to remove members")

        target_membership = await self._repo.get_membership(
            workspace_id, target_user_id
        )
        if target_membership is None:
            raise NotFoundError("User is not a member of this workspace")

        # Prevent privilege escalation: a role can only remove members of equal
        # or lower rank (e.g. ADMIN cannot remove an OWNER).
        if ROLE_RANK[target_membership.role] > ROLE_RANK[actor_membership.role]:
            raise ForbiddenError("Cannot remove a member with a higher role")

        if target_membership.role == WorkspaceRole.OWNER:
            owner_count = await self._repo.count_owners_for_update(workspace_id)
            if owner_count <= 1:
                raise ConflictError("Cannot remove the last owner of a workspace")

        await self._repo.remove_member(workspace_id, target_user_id)

    async def get_user_role(
        self, actor: User, workspace_id: uuid.UUID
    ) -> WorkspaceRole:
        membership = await self._repo.get_membership(workspace_id, actor.id)
        if membership is None:
            raise ForbiddenError("Not a member of this workspace")
        return membership.role

    async def list_members(
        self, actor: User, workspace_id: uuid.UUID
    ) -> list[WorkspaceMember]:
        membership = await self._repo.get_membership(workspace_id, actor.id)
        if membership is None:
            raise ForbiddenError("Not a member of this workspace")
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
