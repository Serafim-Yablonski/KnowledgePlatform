from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from src.domain.documents import ContentType, DocumentStatus, FailureReason
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
        # The workers' claim (ADR-027): only rows owed work.
        sa.Index(
            "ix_documents_owed_available_at",
            "available_at",
            postgresql_where=sa.text("status IN ('PENDING', 'PROCESSING', 'READY')"),
        ),
        # Stored form: enum names, not values (SQLAlchemy's default for sa.Enum).
        sa.CheckConstraint(
            "failure_reason IS NULL OR status = 'FAILED'",
            name="ck_documents_failure_reason_only_when_failed",
        ),
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
    # The name on the uploader's machine; None if it had none or predates the column.
    original_filename: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, native_enum=False, length=20),
        nullable=False,
        server_default=sa.text("'PENDING'"),
    )
    # Why it is FAILED; None otherwise, and on rows that failed before the column.
    failure_reason: Mapped[FailureReason | None] = mapped_column(
        sa.Enum(FailureReason, native_enum=False, length=20), nullable=True
    )
    # The work queue (ADR-027): claims in the current phase, and when the row may be
    # claimed next (a lease deadline, or a backoff). An instant: compared with now().
    attempts: Mapped[int] = mapped_column(
        sa.SmallInteger, nullable=False, server_default=sa.text("0")
    )
    available_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )
    # The upload's trace context (a propagator carrier); each attempt's span starts
    # under it. Written only by our API, which ignores a client's trace context.
    trace_context: Mapped[dict[str, str] | None] = mapped_column(
        postgresql.JSONB, nullable=True
    )
