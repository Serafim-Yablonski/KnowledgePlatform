"""drop ix_chunks_embedding_hnsw: search is exact per workspace (ADR-029)

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
Create Date: 2026-11-03 00:00:00.000000

No search used it (EXPLAIN: the query never had the shape an HNSW scan serves), yet
every chunk insert maintained it. Shared by all workspaces and filtered after the
scan, it would also return fewer than top_k results for a workspace whose chunks
are a small share of the table.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c0d1e2f3a4b5"
down_revision: str | None = "b9c0d1e2f3a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")


def downgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
