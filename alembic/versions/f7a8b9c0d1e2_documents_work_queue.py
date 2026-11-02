"""documents work queue: attempts and available_at (ADR-027)

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-11-02 00:00:00.000000

Workers claim owed documents from this table. attempts counts claims in the current
phase; available_at is when a row may be claimed next: a lease deadline while a worker
holds it, a backoff after a transient failure. Existing rows get now(), so documents
owed work before the upgrade are claimable at once.

available_at is TIMESTAMPTZ although created_at is naive: the claim compares it with
now(), and a naive value would be converted through each session's TimeZone.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: str | None = "e6f7a8b9c0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "attempts", sa.SmallInteger(), nullable=False, server_default=sa.text("0")
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # The claim's hot path; only rows owed work are indexed.
    op.create_index(
        "ix_documents_owed_available_at",
        "documents",
        ["available_at"],
        postgresql_where=sa.text("status IN ('PENDING', 'PROCESSING', 'READY')"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_owed_available_at", table_name="documents")
    op.drop_column("documents", "available_at")
    op.drop_column("documents", "attempts")
