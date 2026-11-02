"""one chunk set per document: drop both version columns, unique chunk positions

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-10-24 00:00:00.000000

embed_chunks now replaces a document's whole chunk set in one transaction, under the
document's row lock, so readers never see two sets and no version tag is needed to
tell them apart. UNIQUE (document_id, chunk_index) makes a second set impossible and
takes over both plain indexes: a B-tree serves lookups on any leftmost prefix of its
columns, so document_id alone (reads, the cascade) is covered.

Upgrade fails if a document already has two chunks at one position (a race this
revision fixes); delete and re-upload that document.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: str | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_chunks_document_chunk_index",
        "document_chunks",
        ["document_id", "chunk_index"],
    )
    op.drop_index("ix_chunks_document_version", table_name="document_chunks")
    op.drop_index("ix_chunks_document_id", table_name="document_chunks")
    op.drop_column("document_chunks", "version")
    op.drop_column("documents", "version")


def downgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column("document_chunks", sa.Column("version", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE document_chunks c SET version = d.version "
        "FROM documents d WHERE c.document_id = d.id"
    )
    op.alter_column("document_chunks", "version", nullable=False)
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_chunks_document_version", "document_chunks", ["document_id", "version"]
    )
    op.drop_constraint(
        "uq_chunks_document_chunk_index", "document_chunks", type_="unique"
    )
