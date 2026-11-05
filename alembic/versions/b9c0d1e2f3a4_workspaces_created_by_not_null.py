"""workspaces.created_by NOT NULL, as the model declares

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-11-03 00:00:00.000000

b1c2d3e4f5a6 added the column nullable for rows from before it ("tighten after
backfill"); the tightening never came, while every workspace since has been created
with its creator. Upgrade fails if a workspace still has none ("column contains null
values"): set created_by to its owner's user id, then upgrade again.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b9c0d1e2f3a4"
down_revision: str | None = "a8b9c0d1e2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("workspaces", "created_by", existing_type=sa.UUID(), nullable=False)


def downgrade() -> None:
    op.alter_column("workspaces", "created_by", existing_type=sa.UUID(), nullable=True)
