"""Data migrations, run for real by Alembic on a scratch database.

Sync tests: alembic/env.py calls asyncio.run(), which can't run inside a test's loop.
"""

import uuid
from collections.abc import Generator

import pytest
import sqlalchemy as sa
from testcontainers.postgres import PostgresContainer

from alembic import command
from src.core.config import settings
from tests.support.database import alembic_config, scratch_database

_BEFORE_ORIGINAL_FILENAME = "b3c4d5e6f7a8"
_ORIGINAL_FILENAME = "c4d5e6f7a8b9"
_ONE_CHUNK_SET = "d5e6f7a8b9c0"
_FAILURE_REASON = "e6f7a8b9c0d1"
_WORK_QUEUE = "f7a8b9c0d1e2"


@pytest.fixture
def scratch_db(postgres_container: PostgresContainer) -> Generator[str]:
    with scratch_database(postgres_container, "migration") as url:
        yield url


def test_original_filename_backfill(scratch_db: str) -> None:
    cfg = alembic_config(scratch_db)
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


def _document_with_chunks(
    engine: sa.Engine, *, doc_version: int, chunk_versions: list[int]
) -> uuid.UUID:
    """A document at ``doc_version``, one chunk per entry of ``chunk_versions``;
    each chunk version numbers its positions from 0, as one generation did."""
    ws, user, doc = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    vector = "[" + ",".join(["0.1"] * settings.EMBEDDING_DIMENSIONS) + "]"
    positions: dict[int, int] = {}
    rows = []
    for v in chunk_versions:
        index = positions.get(v, 0)
        positions[v] = index + 1
        rows.append({"doc": doc, "i": index, "v": v, "e": vector})
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, :email, 'x')"
            ),
            {"id": user, "email": f"{user}@example.com"},
        )
        conn.execute(
            sa.text(
                "INSERT INTO workspaces (id, name, slug, created_by) "
                "VALUES (:id, 'W', :slug, :user)"
            ),
            {"id": ws, "slug": f"w-{ws.hex[:8]}", "user": user},
        )
        conn.execute(
            sa.text(
                "INSERT INTO documents (id, workspace_id, title, content_type, "
                "file_path, file_size_bytes, uploaded_by, version) "
                "VALUES (:id, :ws, 'T', 'PDF', '/x.pdf', 1, :user, :v)"
            ),
            {"id": doc, "ws": ws, "user": user, "v": doc_version},
        )
        conn.execute(
            sa.text(
                "INSERT INTO document_chunks (document_id, chunk_index, text, "
                "embedding, token_count, version) "
                "VALUES (:doc, :i, 'c', CAST(:e AS vector), 1, :v)"
            ),
            rows,
        )
    return doc


def test_one_chunk_set_drops_both_versions_and_keeps_the_chunks(
    scratch_db: str,
) -> None:
    cfg = alembic_config(scratch_db)
    command.upgrade(cfg, _ORIGINAL_FILENAME)
    engine = sa.create_engine(scratch_db)
    # Renamed before step 4: the document's version moved past its chunks'.
    doc = _document_with_chunks(engine, doc_version=3, chunk_versions=[1, 1, 1])

    command.upgrade(cfg, _ONE_CHUNK_SET)

    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        doc_columns = {c["name"] for c in inspector.get_columns("documents")}
        chunk_columns = {c["name"] for c in inspector.get_columns("document_chunks")}
        uniques = inspector.get_unique_constraints("document_chunks")
        indexes = {i["name"] for i in inspector.get_indexes("document_chunks")}
        positions = conn.scalars(
            sa.text(
                "SELECT chunk_index FROM document_chunks WHERE document_id = :d "
                "ORDER BY chunk_index"
            ),
            {"d": doc},
        ).all()
    assert "version" not in doc_columns | chunk_columns
    assert [(u["name"], u["column_names"]) for u in uniques] == [
        ("uq_chunks_document_chunk_index", ["document_id", "chunk_index"])
    ]
    assert indexes == {"uq_chunks_document_chunk_index", "ix_chunks_embedding_hnsw"}
    assert positions == [0, 1, 2]

    command.downgrade(cfg, _ORIGINAL_FILENAME)
    with engine.connect() as conn:
        versions = conn.execute(
            sa.text(
                "SELECT d.version, c.version FROM document_chunks c "
                "JOIN documents d ON d.id = c.document_id"
            )
        ).all()
    assert versions == [(1, 1)] * 3
    engine.dispose()


def test_one_chunk_set_refuses_a_document_with_two_sets(scratch_db: str) -> None:
    cfg = alembic_config(scratch_db)
    command.upgrade(cfg, _ORIGINAL_FILENAME)
    engine = sa.create_engine(scratch_db)
    # Two sets at positions 0 and 1 (two generations, or the C2 race): no repair.
    _document_with_chunks(engine, doc_version=2, chunk_versions=[1, 1, 2, 2])
    engine.dispose()

    with pytest.raises(sa.exc.IntegrityError, match="uq_chunks_document_chunk_index"):
        command.upgrade(cfg, _ONE_CHUNK_SET)


def _status_default(conn: sa.Connection) -> str | None:
    columns = sa.inspect(conn).get_columns("documents")
    return next(c["default"] for c in columns if c["name"] == "status")


def test_failure_reason_adds_the_column_the_check_and_a_loadable_default(
    scratch_db: str,
) -> None:
    cfg = alembic_config(scratch_db)
    command.upgrade(cfg, _ONE_CHUNK_SET)
    engine = sa.create_engine(scratch_db)
    ws, user, failed, defaulted = (uuid.uuid4() for _ in range(4))
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, 'r@example.com', 'x')"
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
        insert = (
            "INSERT INTO documents (id, workspace_id, title, content_type, "
            "file_path, file_size_bytes, uploaded_by{col}) "
            "VALUES (:id, :ws, 'T', 'PDF', '/x.pdf', 1, :user{val})"
        )
        conn.execute(
            sa.text(insert.format(col=", status", val=", 'FAILED'")),
            {"id": failed, "ws": ws, "user": user},
        )
        # Left to the old default: lowercase, which the ORM can't load.
        conn.execute(
            sa.text(insert.format(col="", val="")),
            {"id": defaulted, "ws": ws, "user": user},
        )
        assert (
            conn.scalar(
                sa.text("SELECT status FROM documents WHERE id = :id"),
                {"id": defaulted},
            )
            == "pending"
        )

    command.upgrade(cfg, _FAILURE_REASON)

    with engine.connect() as conn:
        stored = {
            row.id: (row.status, row.failure_reason)
            for row in conn.execute(
                sa.text("SELECT id, status, failure_reason FROM documents")
            )
        }
        checks = {
            c["name"] for c in sa.inspect(conn).get_check_constraints("documents")
        }
        default = _status_default(conn)
    assert stored == {failed: ("FAILED", None), defaulted: ("PENDING", None)}
    assert default is not None and "'PENDING'" in default
    assert "ck_documents_failure_reason_only_when_failed" in checks
    with (
        engine.begin() as conn,
        pytest.raises(
            sa.exc.IntegrityError, match="ck_documents_failure_reason_only_when_failed"
        ),
    ):
        conn.execute(
            sa.text("UPDATE documents SET failure_reason = 'NO_TEXT' WHERE id = :id"),
            {"id": defaulted},
        )

    command.downgrade(cfg, _ONE_CHUNK_SET)
    with engine.connect() as conn:
        columns = {c["name"] for c in sa.inspect(conn).get_columns("documents")}
        default = _status_default(conn)
    assert "failure_reason" not in columns
    assert default is not None and "'pending'" in default
    engine.dispose()


def test_work_queue_columns_make_owed_rows_claimable_at_once(scratch_db: str) -> None:
    cfg = alembic_config(scratch_db)
    command.upgrade(cfg, _FAILURE_REASON)
    engine = sa.create_engine(scratch_db)
    ws, user = uuid.uuid4(), uuid.uuid4()
    statuses = ["PENDING", "PROCESSING", "READY", "INDEXED", "FAILED"]
    ids = {status: uuid.uuid4() for status in statuses}
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password) "
                "VALUES (:id, 'q@example.com', 'x')"
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
                "file_path, file_size_bytes, uploaded_by, status) "
                "VALUES (:id, :ws, 'T', 'PDF', '/x.pdf', 1, :user, :status)"
            ),
            [{"id": ids[s], "ws": ws, "user": user, "status": s} for s in statuses],
        )

    command.upgrade(cfg, _WORK_QUEUE)

    with engine.connect() as conn:
        # Rows owed work before the upgrade are claimable the moment workers start.
        stored = {
            row.id: (row.attempts, row.due)
            for row in conn.execute(
                sa.text(
                    "SELECT id, attempts, available_at <= now() AS due FROM documents"
                )
            )
        }
        types = {
            c["name"]: c["type"] for c in sa.inspect(conn).get_columns("documents")
        }
        index = conn.execute(
            sa.text(
                "SELECT pg_get_indexdef(i.indexrelid, 1, true) AS col, "
                "pg_get_expr(i.indpred, i.indrelid) AS pred "
                "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE c.relname = 'ix_documents_owed_available_at'"
            )
        ).one()
    assert stored == {ids[s]: (0, True) for s in statuses}
    # An instant, not a wall-clock reading: the lease compares against now().
    assert isinstance(types["available_at"], sa.TIMESTAMP)
    assert types["available_at"].timezone
    assert index.col == "available_at"
    for status in ("PENDING", "PROCESSING", "READY"):
        assert f"'{status}'" in index.pred
    for status in ("INDEXED", "FAILED"):
        assert f"'{status}'" not in index.pred

    command.downgrade(cfg, _FAILURE_REASON)
    with engine.connect() as conn:
        columns = {c["name"] for c in sa.inspect(conn).get_columns("documents")}
        indexes = {i["name"] for i in sa.inspect(conn).get_indexes("documents")}
    assert {"attempts", "available_at"}.isdisjoint(columns)
    assert "ix_documents_owed_available_at" not in indexes
    engine.dispose()
