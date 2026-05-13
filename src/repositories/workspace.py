import uuid

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.domain.roles import WorkspaceRole
from src.models.workspace import Workspace, WorkspaceMembership


class SQLAlchemyWorkspaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        slug: str,
        created_by_id: uuid.UUID,
        description: str | None = None,
    ) -> Workspace:
        workspace = Workspace(
            name=name,
            slug=slug,
            description=description,
            created_by=created_by_id,
        )
        self._session.add(workspace)
        # Flush only — the caller (WorkspaceService.create) adds a membership next
        # and the two writes are committed together atomically by add_member().
        await self._session.flush()
        await self._session.refresh(workspace)
        return workspace

    async def get_by_id(self, workspace_id: uuid.UUID) -> Workspace | None:
        result = await self._session.scalars(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        return result.first()

    async def get_by_slug(self, slug: str) -> Workspace | None:
        result = await self._session.scalars(
            select(Workspace).where(Workspace.slug == slug)
        )
        return result.first()

    async def list_for_user(self, user_id: uuid.UUID) -> list[Workspace]:
        # JOIN via membership so only workspaces the user belongs to are returned,
        # ordered by most recently joined first.
        result = await self._session.scalars(
            select(Workspace)
            .join(
                WorkspaceMembership,
                WorkspaceMembership.workspace_id == Workspace.id,
            )
            .where(WorkspaceMembership.user_id == user_id)
            .order_by(WorkspaceMembership.joined_at.desc())
        )
        return list(result.all())

    async def add_member(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: WorkspaceRole,
        invited_by_id: uuid.UUID | None = None,
    ) -> WorkspaceMembership:
        membership = WorkspaceMembership(
            workspace_id=workspace_id,
            user_id=user_id,
            role=role,
            invited_by=invited_by_id,
        )
        self._session.add(membership)
        await self._session.commit()
        await self._session.refresh(membership)
        return membership

    async def remove_member(self, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
        await self._session.execute(
            sa.delete(WorkspaceMembership).where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user_id,
            )
        )
        await self._session.commit()

    async def get_membership(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMembership | None:
        result = await self._session.scalars(
            select(WorkspaceMembership).where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user_id,
            )
        )
        return result.first()

    async def list_members(self, workspace_id: uuid.UUID) -> list[WorkspaceMembership]:
        result = await self._session.scalars(
            select(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == workspace_id)
            .options(selectinload(WorkspaceMembership.user))
            .order_by(WorkspaceMembership.joined_at.asc())
        )
        return list(result.all())

    async def count_members(self, workspace_id: uuid.UUID) -> int:
        result = await self._session.scalar(
            select(func.count())
            .select_from(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == workspace_id)
        )
        return int(result or 0)
