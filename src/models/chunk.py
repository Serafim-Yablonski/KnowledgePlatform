from __future__ import annotations

import uuid

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column

from src.core.config import get_settings
from src.models.base import Base, TimestampMixin

# Captured at import time so the Vector column size is fixed for the process lifetime.
# The migration reads the same Settings value — they MUST agree at deploy time.
# Changing EMBEDDING_DIMENSIONS requires a new migration + full re-embedding.
EMBEDDING_DIMS: int = get_settings().EMBEDDING_DIMENSIONS


class DocumentChunk(TimestampMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        # One chunk set per document, enforced by the database; also the index for
        # reads by document (and the cascade), which use its leftmost prefix.
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_chunk_index"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=sa.text("uuidv7()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMS), nullable=False
    )
    token_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    # "metadata" is a reserved word in SQLAlchemy's DeclarativeBase internals;
    # using metadata_ avoids the collision at the Python attribute level.
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        sa.JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
    )
