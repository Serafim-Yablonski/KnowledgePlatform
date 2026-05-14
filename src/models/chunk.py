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
        # Composite index for re-indexing queries: find all chunks for a document
        # at a specific version to determine which are stale.
        sa.Index("ix_chunks_document_version", "document_id", "version"),
        sa.Index("ix_chunks_document_id", "document_id"),
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
    # HNSW index defined in Alembic migration (not here) because pgvector HNSW
    # requires CREATE INDEX ... USING hnsw syntax not expressible via table args.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMS), nullable=False
    )
    token_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    # "metadata" is a reserved word in SQLAlchemy's DeclarativeBase internals;
    # using metadata_ avoids the collision at the Python attribute level.
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        sa.JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
    )
    # Matches Document.version at the time of chunking. Used to safely delete
    # stale chunks during re-indexing without serving wrong-dimension embeddings.
    version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
