"""document failure_reason, and a status default the ORM can load

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-10-24 00:00:00.000000

failure_reason says why a document is FAILED; a CHECK keeps it on FAILED rows only.
Rows that failed before this revision keep NULL: their reason was never recorded.

SQLAlchemy stores enum names ('PENDING'), but documents.status defaulted to
'pending', which the ORM refuses to load. Every insert set the status, so the default
only reached rows inserted by hand; those are repaired here.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: str | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("failure_reason", sa.String(20), nullable=True)
    )
    op.create_check_constraint(
        "ck_documents_failure_reason_only_when_failed",
        "documents",
        "failure_reason IS NULL OR status = 'FAILED'",
    )
    op.alter_column("documents", "status", server_default=sa.text("'PENDING'"))
    op.execute("UPDATE documents SET status = 'PENDING' WHERE status = 'pending'")


def downgrade() -> None:
    # Repaired rows stay 'PENDING': it is the only form the ORM ever loaded.
    op.alter_column("documents", "status", server_default=sa.text("'pending'"))
    op.drop_constraint(
        "ck_documents_failure_reason_only_when_failed", "documents", type_="check"
    )
    op.drop_column("documents", "failure_reason")
