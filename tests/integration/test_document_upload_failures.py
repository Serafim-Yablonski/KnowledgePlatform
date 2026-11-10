"""Upload failures from the commit point on, against real PostgreSQL.

Once commit() is called its outcome can be unknown, and after it succeeds the row
needs its file: from there on a failure must keep the file (an orphan file is only
garbage; a committed row without its file is corruption).
"""

import io
from pathlib import Path

import pytest
import sqlalchemy as sa
import structlog
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.composition import Services
from src.core import config
from src.domain.documents import DocumentStatus, DocumentUpload
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from tests.support import seed


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config.settings, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


def _upload() -> DocumentUpload:
    return DocumentUpload(
        filename="notes.txt",
        content_type="text/plain",
        declared_size=None,
        stream=io.BytesIO(b"plain text"),
    )


def _files(upload_dir: Path) -> list[Path]:
    return [p for p in upload_dir.rglob("*") if p.is_file()]


async def test_failed_commit_keeps_the_file_and_logs_it(
    committing_engine: AsyncEngine,
    committing_services: Services,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)

    async def connection_lost(self: AsyncSession) -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(AsyncSession, "commit", connection_lost)

    with (
        structlog.testing.capture_logs() as logs,
        pytest.raises(OperationalError),
    ):
        await committing_services.document.create(
            user, ws_id, WorkspaceRole.MEMBER, None, _upload()
        )

    assert len(_files(upload_dir)) == 1
    assert any(e["event"].startswith("upload commit failed") for e in logs)


async def test_an_upload_only_commits_its_row_which_is_claimable_at_once(
    committing_engine: AsyncEngine, committing_services: Services
) -> None:
    """The row is the work owed (ADR-027): nothing is published after the commit,
    so no broker can fail the upload, and a worker may claim the row at once."""
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)

    doc = await committing_services.document.create(
        user, ws_id, WorkspaceRole.MEMBER, None, _upload()
    )

    async with committing_engine.connect() as conn:
        row = (
            await conn.execute(
                sa.select(
                    Document.status,
                    Document.attempts,
                    Document.available_at <= sa.func.now(),
                ).where(Document.id == doc.id)
            )
        ).one()
    assert tuple(row) == (DocumentStatus.PENDING, 0, True)
