from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from src.domain.documents import ContentType, DocumentStatus
from src.models.base import Base, HasIDMixin


class Document(HasIDMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        sa.Index(
            "ix_documents_workspace_created_id",
            "workspace_id",
            "created_at",
            "id",
        ),
        sa.Index("ix_documents_workspace_status", "workspace_id", "status"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    content_type: Mapped[ContentType] = mapped_column(
        sa.Enum(ContentType, native_enum=False, length=20), nullable=False
    )
    raw_text: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    file_path: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, native_enum=False, length=20),
        nullable=False,
        server_default=sa.text("'pending'"),
    )
    version: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("1")
    )
