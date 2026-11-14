"""Deleting a workspace removes its files only after its rows are gone, against real
PostgreSQL. The race tests use separate connections and real commits: row locks are
what make "remove the directory after the delete's commit" safe against uploads.
"""

import io
import uuid
from pathlib import Path
from typing import Any

import anyio
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.composition import Services
from src.core import config
from src.domain.documents import ContentType, DocumentUpload
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.file_store import LocalFileStore
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


async def _workspace(engine: AsyncEngine) -> tuple[User, Workspace]:
    user = await seed.user(engine)
    ws_id = await seed.workspace_of(engine, user)
    async with AsyncSession(engine) as session:
        ws = await session.get_one(Workspace, ws_id)
    return user, ws


async def test_failed_row_delete_keeps_the_files(
    committing_engine: AsyncEngine,
    committing_services: Services,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, ws = await _workspace(committing_engine)
    stored = Path(
        await LocalFileStore(upload_dir).save(
            ws.id, uuid.uuid4(), ContentType.PLAINTEXT, io.BytesIO(b"x")
        )
    )

    async def connection_lost(self: AsyncSession) -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(AsyncSession, "commit", connection_lost)

    with pytest.raises(OperationalError):
        await committing_services.workspace.delete(user, ws, WorkspaceRole.OWNER)

    assert stored.exists()


# ---------------------------------------------------------------------------
# Races between an upload and the workspace's deletion
# ---------------------------------------------------------------------------


async def _document_rows(engine: AsyncEngine, workspace_id: uuid.UUID) -> int:
    async with AsyncSession(engine) as session:
        return (
            await session.scalar(
                sa.select(sa.func.count()).where(Document.workspace_id == workspace_id)
            )
            or 0
        )


async def test_upload_first_the_delete_waits_then_removes_its_file(
    committing_engine: AsyncEngine,
    committing_services: Services,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, ws = await _workspace(committing_engine)
    deleted = anyio.Event()
    reached, release = anyio.Event(), anyio.Event()
    set_file_path = SQLAlchemyDocumentRepository.set_file_path

    async def gated(self: SQLAlchemyDocumentRepository, *args: Any) -> None:
        # Called after the file is written, right before the upload's commit.
        reached.set()
        await release.wait()
        await set_file_path(self, *args)

    monkeypatch.setattr(SQLAlchemyDocumentRepository, "set_file_path", gated)

    async def upload() -> None:
        await committing_services.document.create(
            user, ws.id, WorkspaceRole.MEMBER, None, _upload()
        )

    async def delete() -> None:
        await committing_services.workspace.delete(user, ws, WorkspaceRole.OWNER)
        deleted.set()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(upload)
            await reached.wait()  # row flushed, file written, not committed
            [written] = [p for p in upload_dir.rglob("*") if p.is_file()]
            tg.start_soon(delete)
            await anyio.sleep(0.5)
            assert not deleted.is_set(), "the delete must wait for the upload's lock"
            release.set()

    assert deleted.is_set()
    assert not written.exists()
    assert not (upload_dir / str(ws.id)).exists()
    assert await _document_rows(committing_engine, ws.id) == 0


async def test_delete_first_the_upload_fails_before_writing(
    committing_engine: AsyncEngine,
    committing_services: Services,
    upload_dir: Path,
) -> None:
    user, ws = await _workspace(committing_engine)
    outcome: list[BaseException] = []

    async def upload() -> None:
        try:
            await committing_services.document.create(
                user, ws.id, WorkspaceRole.MEMBER, None, _upload()
            )
        except IntegrityError as exc:
            outcome.append(exc)

    async with AsyncSession(committing_engine) as deleting:
        # The delete holds its row lock, uncommitted, while the upload starts.
        await deleting.execute(sa.delete(Workspace).where(Workspace.id == ws.id))
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                tg.start_soon(upload)
                await anyio.sleep(0.5)
                assert not outcome, "the upload's FK check must wait for the delete"
                assert not any(upload_dir.rglob("*.txt"))
                await deleting.commit()

    assert len(outcome) == 1
    assert not any(upload_dir.rglob("*.txt"))
