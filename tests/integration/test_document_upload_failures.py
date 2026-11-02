"""Upload failures from the commit point on, against real PostgreSQL.

Once commit() is called its outcome can be unknown, and after it succeeds the row
needs its file: from there on a failure must keep the file (an orphan file is only
garbage; a committed row without its file is corruption).
"""

import io
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
import structlog
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import DocumentStatus, DocumentUpload
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.chunk import SQLAlchemyChunkRepository
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.file_store import LocalFileStore
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.services.document import DocumentService


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
async def test_workspace(db_session: AsyncSession, test_user: User) -> Workspace:
    ws = await SQLAlchemyWorkspaceRepository(db_session).create(
        name=f"Upload WS {uuid.uuid4().hex[:6]}",
        slug=f"upload-ws-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await db_session.flush()
    return ws


def _service(session: AsyncSession, root: Path) -> DocumentService:
    return DocumentService(
        SQLAlchemyDocumentRepository(session),
        SQLAlchemyChunkRepository(session),
        session,
        LocalFileStore(root),
    )


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
    db_session: AsyncSession,
    test_workspace: Workspace,
    test_user: User,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def connection_lost() -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(db_session, "commit", connection_lost)

    with (
        structlog.testing.capture_logs() as logs,
        pytest.raises(OperationalError),
    ):
        await _service(db_session, upload_dir).create(
            test_user, test_workspace.id, WorkspaceRole.MEMBER, None, _upload()
        )

    assert len(_files(upload_dir)) == 1
    assert any(e["event"].startswith("upload commit failed") for e in logs)


async def test_an_upload_only_commits_its_row_which_is_claimable_at_once(
    db_session: AsyncSession,
    test_workspace: Workspace,
    test_user: User,
    upload_dir: Path,
) -> None:
    """The row is the work owed (ADR-027): nothing is published after the commit,
    so no broker can fail the upload, and a worker may claim the row at once."""
    doc = await _service(db_session, upload_dir).create(
        test_user, test_workspace.id, WorkspaceRole.MEMBER, None, _upload()
    )

    row = (
        await db_session.execute(
            sa.select(
                Document.status,
                Document.attempts,
                Document.available_at <= sa.func.now(),
            ).where(Document.id == doc.id)
        )
    ).one()
    assert tuple(row) == (DocumentStatus.PENDING, 0, True)
