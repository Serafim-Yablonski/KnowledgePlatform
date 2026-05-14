"""add document_chunks table with HNSW index

Revision ID: e1f2a3b4c5d6
Revises: c1d2e3f4a5b6
Create Date: 2026-05-14 00:00:00.000000

IMPORTANT: The Vector column size is read from Settings.EMBEDDING_DIMENSIONS (default 768).
Changing that value requires a new migration + full re-embedding of all documents.
The HNSW index uses cosine similarity (vector_cosine_ops) — matching the distance
metric used in retrieval queries. HNSW chosen over IVFFlat because:
  - HNSW has better recall at low query latency with no training step.
  - IVFFlat requires building cluster centroids over a representative dataset,
    which is impractical when documents are ingested continuously.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op
from src.core.config import settings

revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Read once at migration-run time so the column size matches the model definition.
_DIMS = settings.EMBEDDING_DIMENSIONS


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            sa.UUID(as_uuid=True),
            server_default=sa.text("uuidv7()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_DIMS), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "metadata_",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_chunks_document_version",
        "document_chunks",
        ["document_id", "version"],
    )
    # HNSW index for approximate nearest-neighbour search over embeddings.
    # m=16 (max connections per layer) and ef_construction=64 (beam width during build)
    # are standard starting values; tune upward if recall degrades at scale.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_index("ix_chunks_document_version", table_name="document_chunks")
    op.drop_index("ix_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
