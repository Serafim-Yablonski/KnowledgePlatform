"""Deleting a workspace removes its files only after its rows are gone, against real
PostgreSQL. The race tests use separate connections and real commits: row locks are
what make "remove the directory after the delete's commit" safe against uploads.
"""

import contextlib
import io
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, InvalidRequestError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.domain.documents import ContentType, DocumentUpload
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.chunk import SQLAlchemyChunkRepository
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.file_store import LocalFileStore
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.services.document import DocumentService
from src.services.workspace import WorkspaceService
from src.workers.tasks.extract_text import extract_text


def _upload() -> DocumentUpload:
    return DocumentUpload(
        filename="notes.txt",
        content_type="text/plain",
        declared_size=None,
        stream=io.BytesIO(b"plain text"),
    )


def _workspaces(session: AsyncSession, root: Path) -> WorkspaceService:
    return WorkspaceService(
        SQLAlchemyWorkspaceRepository(session),
        SQLAlchemyUserRepository(session),
        LocalFileStore(root),
    )


def _documents(session: AsyncSession, root: Path) -> DocumentService:
    return DocumentService(
        SQLAlchemyDocumentRepository(session),
        SQLAlchemyChunkRepository(session),
        session,
        LocalFileStore(root),
    )


async def test_failed_row_delete_keeps_the_files(
    db_session: AsyncSession,
    test_user: User,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = await SQLAlchemyWorkspaceRepository(db_session).create(
        name="Keep WS", slug=f"keep-{uuid.uuid4().hex[:8]}", created_by_id=test_user.id
    )
    stored = Path(
        await LocalFileStore(tmp_path).save(
            ws.id, uuid.uuid4(), ContentType.PLAINTEXT, io.BytesIO(b"x")
        )
    )

    async def connection_lost() -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(db_session, "commit", connection_lost)

    with pytest.raises(OperationalError):
        await _workspaces(db_session, tmp_path).delete(
            test_user, ws, WorkspaceRole.OWNER
        )

    assert stored.exists()


# ---------------------------------------------------------------------------
# Races between an upload and the workspace's deletion (real commits)
# ---------------------------------------------------------------------------


@pytest.fixture
async def committed_workspace(
    test_engine: AsyncEngine,
) -> AsyncGenerator[tuple[User, Workspace]]:
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        user = User(
            email=f"race-{uuid.uuid4().hex[:8]}@example.com", hashed_password="x"
        )
        session.add(user)
        await session.flush()
        ws = Workspace(
            name="Race WS", slug=f"race-{uuid.uuid4().hex[:8]}", created_by=user.id
        )
        session.add(ws)
        await session.commit()
    try:
        yield user, ws
    finally:
        async with AsyncSession(test_engine) as session:
            await session.execute(sa.delete(Workspace).where(Workspace.id == ws.id))
            await session.execute(sa.delete(User).where(User.id == user.id))
            await session.commit()


async def _document_rows(engine: AsyncEngine, workspace_id: uuid.UUID) -> int:
    async with AsyncSession(engine) as session:
        return (
            await session.scalar(
                sa.select(sa.func.count()).where(Document.workspace_id == workspace_id)
            )
            or 0
        )


def _gate_commit(
    session: AsyncSession,
) -> tuple[anyio.Event, anyio.Event, Callable[[], Awaitable[None]]]:
    """Hold the session's commit until ``release`` is set; ``reached`` tells when."""
    reached, release = anyio.Event(), anyio.Event()
    commit = session.commit

    async def gated() -> None:
        reached.set()
        await release.wait()
        await commit()

    return reached, release, gated


async def test_upload_first_the_delete_waits_then_removes_its_file(
    test_engine: AsyncEngine,
    committed_workspace: tuple[User, Workspace],
    tmp_path: Path,
) -> None:
    user, ws = committed_workspace
    deleted = anyio.Event()

    async with (
        AsyncSession(test_engine, expire_on_commit=False) as uploading,
        AsyncSession(test_engine, expire_on_commit=False) as deleting,
    ):
        reached, release, gated = _gate_commit(uploading)
        uploading.commit = gated  # type: ignore[method-assign]

        async def upload() -> None:
            # Committed, then the delete cascades the row away at once, so the
            # upload's refresh after its commit may find nothing: either outcome.
            with contextlib.suppress(InvalidRequestError):
                await _documents(uploading, tmp_path).create(
                    user, ws.id, WorkspaceRole.MEMBER, None, _upload()
                )

        async def delete() -> None:
            await _workspaces(deleting, tmp_path).delete(user, ws, WorkspaceRole.OWNER)
            deleted.set()

        with anyio.fail_after(10), patch.object(extract_text, "delay"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(upload)
                await reached.wait()  # row flushed, file written, not committed
                [written] = [p for p in tmp_path.rglob("*") if p.is_file()]
                tg.start_soon(delete)
                await anyio.sleep(0.5)
                assert not deleted.is_set(), (
                    "the delete must wait for the upload's lock"
                )
                release.set()

    assert deleted.is_set()
    assert not written.exists()
    assert not (tmp_path / str(ws.id)).exists()
    assert await _document_rows(test_engine, ws.id) == 0


async def test_delete_first_the_upload_fails_before_writing(
    test_engine: AsyncEngine,
    committed_workspace: tuple[User, Workspace],
    tmp_path: Path,
) -> None:
    user, ws = committed_workspace
    outcome: list[BaseException] = []

    async def upload(session: AsyncSession) -> None:
        try:
            await _documents(session, tmp_path).create(
                user, ws.id, WorkspaceRole.MEMBER, None, _upload()
            )
        except IntegrityError as exc:
            outcome.append(exc)

    async with (
        AsyncSession(test_engine) as deleting,
        AsyncSession(test_engine) as uploading,
    ):
        # The delete holds its row lock, uncommitted, while the upload starts.
        await deleting.execute(sa.delete(Workspace).where(Workspace.id == ws.id))
        with anyio.fail_after(10), patch.object(extract_text, "delay"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(upload, uploading)
                await anyio.sleep(0.5)
                assert not outcome, "the upload's FK check must wait for the delete"
                assert not tmp_path.exists() or not any(tmp_path.rglob("*.txt"))
                await deleting.commit()

    assert len(outcome) == 1
    assert not any(tmp_path.rglob("*.txt"))
