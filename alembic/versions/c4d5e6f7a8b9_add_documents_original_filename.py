"""add documents.original_filename, backfilled where the old path kept it

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-10-23 00:00:00.000000

Uploads used to be stored at <UPLOAD_DIR>/<workspace>/<document id>/<client name>, so
those rows still carry the uploader's filename in file_path: a row has that layout
exactly when its file's directory is named after its own id. Newer rows
(<workspace>/<document id>.<ext>) and paths outside the layout (rows from the
removed eval fixtures) don't, and stay NULL. So do "upload" (the old fallback for a
nameless file, indistinguishable from a real name) and names the upload rules would
reject. Those rules are copied here, frozen:
importing them from src/ would let a later change rewrite what this migration did.
"""

import unicodedata
from collections.abc import Sequence
from pathlib import PurePosixPath

import sqlalchemy as sa

from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "b3c4d5e6f7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")


def _recovered_name(document_id: str, file_path: str) -> str | None:
    path = PurePosixPath(file_path)
    if path.parent.name != document_id:
        return None
    name = unicodedata.normalize("NFC", path.name).strip()
    if name in ("", ".", "..", "upload") or len(name) > 255:
        return None
    if any(unicodedata.category(c) == "Cc" or c in _BIDI_CONTROLS for c in name):
        return None
    return name


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("original_filename", sa.String(255), nullable=True)
    )
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, file_path FROM documents")).all()
    updates = [
        {"id": doc_id, "name": name}
        for doc_id, file_path in rows
        if (name := _recovered_name(str(doc_id), file_path)) is not None
    ]
    if updates:
        conn.execute(
            sa.text("UPDATE documents SET original_filename = :name WHERE id = :id"),
            updates,
        )


def downgrade() -> None:
    op.drop_column("documents", "original_filename")
