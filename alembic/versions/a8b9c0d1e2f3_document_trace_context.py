"""documents.trace_context: the upload's trace, continued by the workers (ADR-027)

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-11-02 00:00:00.000000

The W3C trace context of the span the upload ran in, as a propagator carrier
(``traceparent`` and any ``tracestate``). Each pipeline attempt starts its span under
it, so one trace shows an upload through to INDEXED or FAILED. NULL on rows from
before the column: their attempts start traces of their own.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: str | None = "f7a8b9c0d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("trace_context", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "trace_context")
