"""add workspace columns and memberships table

Revision ID: b1c2d3e4f5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-05-13 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Truncate any oversized rows before narrowing to avoid a hard abort on ALTER.
    op.execute("UPDATE workspaces SET name = LEFT(name, 100) WHERE length(name) > 100")
    # Narrow name column from 255 → 100 chars to match domain constraint.
    op.alter_column(
        "workspaces",
        "name",
        existing_type=sa.String(255),
        type_=sa.String(100),
        existing_nullable=False,
    )
    op.execute("UPDATE workspaces SET slug = LEFT(slug, 55) WHERE length(slug) > 55")
    # Narrow slug column to match max slug length (50 chars + hyphen + 4 suffix).
    op.alter_column(
        "workspaces",
        "slug",
        existing_type=sa.String(255),
        type_=sa.String(55),
        existing_nullable=False,
    )
    op.add_column(
        "workspaces",
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column(
            "created_by",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=True,  # nullable for existing rows; tighten after backfill
        ),
    )
    op.add_column(
        "workspaces",
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
    )
    op.create_index("ix_workspaces_slug", "workspaces", ["slug"])

    op.create_table(
        "workspace_memberships",
        sa.Column(
            "workspace_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.String(20),
            nullable=False,
        ),
        sa.Column(
            "invited_by",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "joined_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("workspace_id", "user_id"),
    )


def downgrade() -> None:
    op.drop_table("workspace_memberships")
    op.drop_index("ix_workspaces_slug", table_name="workspaces")
    op.drop_column("workspaces", "is_active")
    op.drop_column("workspaces", "created_by")
    op.drop_column("workspaces", "description")
    op.alter_column(
        "workspaces",
        "slug",
        existing_type=sa.String(55),
        type_=sa.String(255),
        existing_nullable=False,
    )
    op.alter_column(
        "workspaces",
        "name",
        existing_type=sa.String(100),
        type_=sa.String(255),
        existing_nullable=False,
    )
