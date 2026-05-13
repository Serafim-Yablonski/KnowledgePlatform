from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.domain.roles import WorkspaceRole
from src.models.base import Base, HasIDMixin

if TYPE_CHECKING:
    from src.models.user import User


class Workspace(HasIDMixin, Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        sa.UniqueConstraint("slug", name="uq_workspaces_slug"),
        sa.Index("ix_workspaces_slug", "slug"),
    )

    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    slug: Mapped[str] = mapped_column(sa.String(55), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_by: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true")
    )

    # lazy="raise" prevents accidental N+1 queries: any code that loads Workspace
    # without explicitly calling selectinload(Workspace.members) will raise at
    # runtime instead of silently issuing extra queries per row.
    members: Mapped[list[WorkspaceMembership]] = relationship(
        "WorkspaceMembership",
        back_populates="workspace",
        lazy="raise",
    )


class WorkspaceMembership(Base):
    __tablename__ = "workspace_memberships"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[WorkspaceRole] = mapped_column(
        sa.Enum(WorkspaceRole, native_enum=False, length=20),
        nullable=False,
    )
    invited_by: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    joined_at: Mapped[datetime] = mapped_column(
        server_default=sa.text("now()"), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(
        "Workspace",
        back_populates="members",
        lazy="raise",
    )
    user: Mapped[User] = relationship(
        "User",
        foreign_keys=[user_id],
        lazy="raise",
    )
