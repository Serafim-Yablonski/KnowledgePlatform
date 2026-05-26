"""Unit tests for DocumentService using in-memory Protocol stubs."""

import io
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import get_settings
from src.core.exceptions import ForbiddenError, InputValidationError, NotFoundError
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    ContentType,
    Cursor,
    DocumentStatus,
    DocumentUpdateInput,
)
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.services.document import DocumentService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> User:
    u = User()
    u.id = uuid.uuid4()
    u.email = "user@example.com"
    u.hashed_password = "hashed"
    u.display_name = None
    u.is_active = True
    u.created_at = datetime.now(UTC).replace(tzinfo=None)
    u.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return u


def _make_workspace() -> Workspace:
    ws = Workspace()
    ws.id = uuid.uuid4()
    ws.name = "Test WS"
    ws.slug = "test-ws-abcd"
    ws.description = None
    ws.created_by = uuid.uuid4()
    ws.is_active = True
    ws.created_at = datetime.now(UTC).replace(tzinfo=None)
    ws.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return ws


def _make_document(workspace_id: uuid.UUID, uploaded_by: uuid.UUID) -> Document:
    doc = Document()
    doc.id = uuid.uuid4()
    doc.workspace_id = workspace_id
    doc.title = "Test Doc"
    doc.content_type = ContentType.PDF
    doc.raw_text = None
    doc.file_path = "/data/uploads/test/file.pdf"
    doc.file_size_bytes = 1024
    doc.uploaded_by = uploaded_by
    doc.status = DocumentStatus.PENDING
    doc.version = 1
    doc.created_at = datetime.now(UTC).replace(tzinfo=None)
    doc.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return doc


_MIME_DEFAULT_CONTENT: dict[str, bytes] = {
    "application/pdf": b"%PDF-1.4 fake content for tests",
    "text/plain": b"Hello, this is plain text.",
    "text/markdown": b"# Hello\nThis is markdown.",
}


def _make_upload_file(
    *,
    size: int = 1024,
    content_type: str = "application/pdf",
    filename: str = "test.pdf",
    content: bytes | None = None,
) -> Any:
    if content is None:
        content = _MIME_DEFAULT_CONTENT.get(content_type, b"fake content")
    f = MagicMock()
    f.size = size
    f.content_type = content_type
    f.filename = filename
    f.file = io.BytesIO(content)
    f.read = AsyncMock(return_value=content[:8])
    f.seek = AsyncMock()
    return f


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubDocumentRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, Document] = {}

    async def create(
        self,
        workspace_id: uuid.UUID,
        title: str,
        content_type: ContentType,
        file_path: str,
        file_size_bytes: int,
        uploaded_by: uuid.UUID,
    ) -> Document:
        doc = Document()
        doc.id = uuid.uuid4()
        doc.workspace_id = workspace_id
        doc.title = title
        doc.content_type = content_type
        doc.raw_text = None
        doc.file_path = file_path
        doc.file_size_bytes = file_size_bytes
        doc.uploaded_by = uploaded_by
        doc.status = DocumentStatus.PENDING
        doc.version = 1
        doc.created_at = datetime.now(UTC).replace(tzinfo=None)
        doc.updated_at = datetime.now(UTC).replace(tzinfo=None)
        self._store[doc.id] = doc
        return doc

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return self._store.get(document_id)

    async def update(self, document: Document, data: DocumentUpdateInput) -> Document:
        if data.title is not None:
            document.title = data.title
        document.version += 1
        return document

    async def delete(self, document: Document) -> None:
        self._store.pop(document.id, None)

    async def list_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        limit: int = 20,
        cursor: Cursor | None = None,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], Cursor | None]:
        docs = [d for d in self._store.values() if d.workspace_id == workspace_id]
        if status is not None:
            docs = [d for d in docs if d.status == status]
        return docs, None


class MockSession:
    """Minimal async session stub — satisfies service.commit/refresh calls."""

    async def commit(self) -> None:
        pass

    async def refresh(self, obj: Any) -> None:
        pass


def _make_service() -> tuple[DocumentService, StubDocumentRepository]:
    repo = StubDocumentRepository()
    service = DocumentService(repo=repo, session=MockSession())  # type: ignore[arg-type]
    return service, repo


# ---------------------------------------------------------------------------
# create — validation
# ---------------------------------------------------------------------------


async def test_file_too_large_raises_validation_error(tmp_path: Path) -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    upload = _make_upload_file(size=get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)

    with pytest.raises(InputValidationError, match="exceeds maximum size"):
        await service.create(user, workspace, WorkspaceRole.MEMBER, "My Doc", upload)


async def test_unsupported_mime_type_raises_validation_error() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    upload = _make_upload_file(content_type="image/png", filename="photo.png")

    with pytest.raises(InputValidationError, match="Unsupported file type"):
        await service.create(user, workspace, WorkspaceRole.MEMBER, "My Doc", upload)


async def test_allowed_mime_types_are_accepted(tmp_path: Path) -> None:
    for mime, expected_ct in ALLOWED_CONTENT_TYPES.items():
        service, repo = _make_service()
        user = _make_user()
        workspace = _make_workspace()
        upload = _make_upload_file(content_type=mime, filename="file.txt")
        import src.services.document as svc_mod

        original = svc_mod.settings.UPLOAD_DIR
        svc_mod.settings.UPLOAD_DIR = str(tmp_path)  # type: ignore[assignment]
        try:
            result = await service.create(
                user, workspace, WorkspaceRole.MEMBER, "Title", upload
            )
            assert result.content_type == expected_ct
        finally:
            svc_mod.settings.UPLOAD_DIR = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# create/update/delete — role authorization
# ---------------------------------------------------------------------------


async def test_viewer_cannot_create_document() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    upload = _make_upload_file()

    with pytest.raises(ForbiddenError):
        await service.create(user, workspace, WorkspaceRole.VIEWER, "Doc", upload)


async def test_viewer_cannot_update_document() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    doc = _make_document(workspace.id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.update(
            user,
            workspace,
            WorkspaceRole.VIEWER,
            doc.id,
            DocumentUpdateInput(title="X"),
        )


async def test_viewer_cannot_delete_document() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    doc = _make_document(workspace.id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.delete(user, workspace, WorkspaceRole.VIEWER, doc.id)


async def test_member_cannot_delete_document() -> None:
    """MEMBER role lacks delete_document permission."""
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    doc = _make_document(workspace.id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.delete(user, workspace, WorkspaceRole.MEMBER, doc.id)


# ---------------------------------------------------------------------------
# delete — tolerates missing file on disk
# ---------------------------------------------------------------------------


async def test_delete_handles_missing_file_gracefully() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()

    doc = _make_document(workspace.id, user.id)
    doc.file_path = "/nonexistent/path/that/does/not/exist.pdf"
    repo._store[doc.id] = doc

    # Must not raise even though the file doesn't exist
    await service.delete(user, workspace, WorkspaceRole.OWNER, doc.id)

    assert await repo.get_by_id(doc.id) is None


async def test_delete_nonexistent_document_raises_not_found() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace = _make_workspace()

    with pytest.raises(NotFoundError):
        await service.delete(user, workspace, WorkspaceRole.OWNER, uuid.uuid4())


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


async def test_get_wrong_workspace_raises_not_found() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()
    other_workspace = _make_workspace()

    doc = _make_document(other_workspace.id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(NotFoundError):
        await service.get(user, workspace, doc.id)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


async def test_update_increments_version() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace = _make_workspace()

    doc = _make_document(workspace.id, user.id)
    repo._store[doc.id] = doc
    original_version = doc.version

    result = await service.update(
        user, workspace, WorkspaceRole.MEMBER, doc.id, DocumentUpdateInput(title="New")
    )
    assert result.version == original_version + 1
    assert result.title == "New"
