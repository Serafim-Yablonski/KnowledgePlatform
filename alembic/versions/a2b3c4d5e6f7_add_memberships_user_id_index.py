"""add index on workspace_memberships.user_id

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-06-01 00:00:00.000000

Without this index, any query filtering workspace_memberships by user_id alone
(list_for_user, get_workspace_for_user) performs a full table scan because the
composite PK (workspace_id, user_id) cannot be used when user_id is not the
leading column.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_memberships_user_id",
        "workspace_memberships",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_memberships_user_id", table_name="workspace_memberships")
