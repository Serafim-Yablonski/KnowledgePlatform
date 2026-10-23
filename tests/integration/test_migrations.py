"""Data migrations, run for real by Alembic on a scratch database.

Sync tests: alembic/env.py calls asyncio.run(), which can't run inside a test's loop.
"""

import uuid
from collections.abc import Generator

import pytest
import sqlalchemy as sa
from alembic.config import Config
from testcontainers.postgres import PostgresContainer

from alembic import command

_BEFORE_ORIGINAL_FILENAME = "b3c4d5e6f7a8"
_ORIGINAL_FILENAME = "c4d5e6f7a8b9"


@pytest.fixture
def scratch_db(postgres_container: PostgresContainer) -> Generator[str]:
    """A fresh database in the session's container; yields its psycopg URL."""
    base = sa.make_url(postgres_container.get_connection_url())
    name = f"migration_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(
        base.set(drivername="postgresql+psycopg"), isolation_level="AUTOCOMMIT"
    )
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    try:
        yield base.set(drivername="postgresql+psycopg", database=name).render_as_string(
            hide_password=False
        )
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _alembic(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option(
        "sqlalchemy.url", url.replace("postgresql+psycopg", "postgresql+asyncpg")
    )
    return cfg


def test_original_filename_backfill(scratch_db: str) -> None:
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, _BEFORE_ORIGINAL_FILENAME)
    engine = sa.create_engine(scratch_db)
    ws, user = uuid.uuid4(), uuid.uuid4()
    keys = ["old_real", "old_fallback", "old_unsafe", "new_layout", "outside_layout"]
    ids = {key: uuid.uuid4() for key in keys}
    paths = {
        # Before step 2: <UPLOAD_DIR>/<ws>/<doc_id>/<client name>. This one as macOS
        # sends it (NFD); the backfill normalizes it to NFC.
        "old_real": f"/data/uploads/{ws}/{ids['old_real']}/re\u0301sume\u0301.pdf",
        "old_fallback": f"/data/uploads/{ws}/{ids['old_fallback']}/upload",
        "old_unsafe": f"/data/uploads/{ws}/{ids['old_unsafe']}/a\tb.pdf",
        # After step 2 the name was never kept. A path outside the upload layout:
        # rows written by the removed eval fixtures (ADR-024) may still exist.
        "new_layout": f"/data/uploads/{ws}/{ids['new_layout']}.pdf",
        "outside_layout": "/app/evals/test_documents/postgres-indexing.md",
    }
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, 'm@example.com', 'x')"
            ),
            {"id": user},
        )
        conn.execute(
            sa.text(
                "INSERT INTO workspaces (id, name, slug, created_by) "
                "VALUES (:id, 'W', 'w', :user)"
            ),
            {"id": ws, "user": user},
        )
        conn.execute(
            sa.text(
                "INSERT INTO documents (id, workspace_id, title, content_type, "
                "file_path, file_size_bytes, uploaded_by) "
                "VALUES (:id, :ws, 'T', 'PDF', :path, 1, :user)"
            ),
            [
                {"id": ids[key], "ws": ws, "path": paths[key], "user": user}
                for key in keys
            ],
        )

    command.upgrade(cfg, _ORIGINAL_FILENAME)

    with engine.connect() as conn:
        stored = dict(
            conn.execute(sa.text("SELECT id, original_filename FROM documents")).all()
        )
    assert {key: stored[ids[key]] for key in keys} == {
        "old_real": "résumé.pdf",
        "old_fallback": None,
        "old_unsafe": None,
        "new_layout": None,
        "outside_layout": None,
    }

    command.downgrade(cfg, _BEFORE_ORIGINAL_FILENAME)
    with engine.connect() as conn:
        columns = {c["name"] for c in sa.inspect(conn).get_columns("documents")}
    assert "original_filename" not in columns
    engine.dispose()
