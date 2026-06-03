"""add index on documents.uploaded_by

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-06-02 00:00:00.000000

PostgreSQL requires an index on FK columns to avoid full table scans during
cascading deletes on users and for any future "documents by uploader" query.
Every other FK in the schema is indexed; this brings uploaded_by into parity.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: str | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_documents_uploaded_by", "documents", ["uploaded_by"])


def downgrade() -> None:
    op.drop_index("ix_documents_uploaded_by", table_name="documents")
