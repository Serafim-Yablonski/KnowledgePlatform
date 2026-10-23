"""Unit tests for DocumentService using in-memory Protocol stubs."""

import io
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.core.config import get_settings
from src.core.exceptions import (
    ConflictError,
    ForbiddenError,
    InputValidationError,
    NotFoundError,
)
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    ChunkText,
    ContentType,
    Cursor,
    DocumentStatus,
    DocumentUpdateInput,
    DocumentUpload,
)
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.services.document import (
    FAILED_DETAIL,
    INVALID_CURSOR_DETAIL,
    NOT_INDEXED_DETAIL,
    DocumentService,
)

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
    size: int | None = 1024,
    content_type: str = "application/pdf",
    filename: str = "test.pdf",
    content: bytes | None = None,
) -> DocumentUpload:
    if content is None:
        content = _MIME_DEFAULT_CONTENT.get(content_type, b"fake content")
    return DocumentUpload(
        filename=filename,
        content_type=content_type,
        declared_size=size,
        stream=io.BytesIO(content),
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubDocumentRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, Document] = {}
        # list_by_workspace returns next_cursor and records the cursor it was given.
        self.next_cursor: Cursor | None = None
        self.received_cursor: Cursor | None = None

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

    async def get_by_id(
        self, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> Document | None:
        doc = self._store.get(document_id)
        return doc if doc is not None and doc.workspace_id == workspace_id else None

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
        self.received_cursor = cursor
        docs = [d for d in self._store.values() if d.workspace_id == workspace_id]
        if status is not None:
            docs = [d for d in docs if d.status == status]
        return docs, self.next_cursor


class StubChunkRepository:
    """In memory, filtered like the real query: document, version, start, order."""

    def __init__(self) -> None:
        self.rows: list[tuple[uuid.UUID, int, ChunkText]] = []
        self.calls = 0

    def add(self, doc: Document, version: int, *indexes: int) -> None:
        for i in indexes:
            text = ChunkText(chunk_index=i, section=None, chunk_text=f"v{version} {i}")
            self.rows.append((doc.id, version, text))

    def _of(self, document_id: uuid.UUID, version: int) -> list[ChunkText]:
        return sorted(
            (c for d, v, c in self.rows if d == document_id and v == version),
            key=lambda c: c.chunk_index,
        )

    async def read_page(
        self, document_id: uuid.UUID, *, version: int, start: int, limit: int
    ) -> list[ChunkText]:
        self.calls += 1
        chunks = self._of(document_id, version)
        return [c for c in chunks if c.chunk_index >= start][:limit]

    async def count(self, document_id: uuid.UUID, *, version: int) -> int:
        self.calls += 1
        return len(self._of(document_id, version))


class MockSession:
    """Minimal async session stub — satisfies service.commit/refresh calls."""

    async def commit(self) -> None:
        pass

    async def refresh(self, obj: Any) -> None:
        pass


def _make_service() -> tuple[DocumentService, StubDocumentRepository]:
    repo = StubDocumentRepository()
    service = DocumentService(
        repo=repo,
        chunk_repo=StubChunkRepository(),
        session=MockSession(),  # type: ignore[arg-type]
    )
    return service, repo


# ---------------------------------------------------------------------------
# create — validation
# ---------------------------------------------------------------------------


async def test_file_too_large_raises_validation_error(tmp_path: Path) -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(size=get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)

    with pytest.raises(InputValidationError, match="exceeds maximum size"):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "My Doc", upload)


async def test_unsupported_mime_type_raises_validation_error() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(content_type="image/png", filename="photo.png")

    with pytest.raises(InputValidationError, match="Unsupported file type"):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "My Doc", upload)


async def test_allowed_mime_types_are_accepted(tmp_path: Path) -> None:
    for mime, expected_ct in ALLOWED_CONTENT_TYPES.items():
        service, repo = _make_service()
        user = _make_user()
        workspace_id = uuid.uuid4()
        upload = _make_upload_file(content_type=mime, filename="file.txt")
        import src.services.document as svc_mod

        original = svc_mod.settings.UPLOAD_DIR
        svc_mod.settings.UPLOAD_DIR = str(tmp_path)  # type: ignore[assignment]
        try:
            result = await service.create(
                user, workspace_id, WorkspaceRole.MEMBER, "Title", upload
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
    workspace_id = uuid.uuid4()
    upload = _make_upload_file()

    with pytest.raises(ForbiddenError):
        await service.create(user, workspace_id, WorkspaceRole.VIEWER, "Doc", upload)


async def test_viewer_cannot_update_document() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.update(
            user,
            workspace_id,
            WorkspaceRole.VIEWER,
            doc.id,
            DocumentUpdateInput(title="X"),
        )


async def test_viewer_cannot_delete_document() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.delete(user, workspace_id, WorkspaceRole.VIEWER, doc.id)


async def test_member_cannot_delete_document() -> None:
    """MEMBER role lacks delete_document permission."""
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(ForbiddenError):
        await service.delete(user, workspace_id, WorkspaceRole.MEMBER, doc.id)


# ---------------------------------------------------------------------------
# delete — tolerates missing file on disk
# ---------------------------------------------------------------------------


async def test_delete_handles_missing_file_gracefully() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()

    doc = _make_document(workspace_id, user.id)
    doc.file_path = "/nonexistent/path/that/does/not/exist.pdf"
    repo._store[doc.id] = doc

    # Must not raise even though the file doesn't exist
    await service.delete(user, workspace_id, WorkspaceRole.OWNER, doc.id)

    assert doc.id not in repo._store


async def test_delete_nonexistent_document_raises_not_found() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()

    with pytest.raises(NotFoundError):
        await service.delete(user, workspace_id, WorkspaceRole.OWNER, uuid.uuid4())


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


async def test_get_wrong_workspace_raises_not_found() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    other_workspace_id = uuid.uuid4()

    doc = _make_document(other_workspace_id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(NotFoundError):
        await service.get(workspace_id, doc.id)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


async def test_update_increments_version() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()

    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc
    original_version = doc.version

    result = await service.update(
        user,
        workspace_id,
        WorkspaceRole.MEMBER,
        doc.id,
        DocumentUpdateInput(title="New"),
    )
    assert result.version == original_version + 1
    assert result.title == "New"


# ---------------------------------------------------------------------------
# _validate_magic_bytes — binary content in text file
# ---------------------------------------------------------------------------


async def test_validate_magic_bytes_wrong_pdf_header_raises() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(
        content_type="application/pdf",
        content=b"not a real pdf file",
    )
    with pytest.raises(InputValidationError, match="does not match"):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "Doc", upload)


async def test_validate_magic_bytes_binary_in_text_file_raises() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(
        content_type="text/plain",
        content=b"\x80\x81\x82\x83",  # invalid UTF-8 continuation bytes
    )

    with pytest.raises(InputValidationError, match="does not match"):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "Doc", upload)


# ---------------------------------------------------------------------------
# create — path traversal guard
# ---------------------------------------------------------------------------


async def test_create_path_traversal_filename_raises() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(content_type="text/plain", filename="safe.txt")

    with (
        patch("pathlib.Path.is_relative_to", return_value=False),
        pytest.raises(InputValidationError, match="Invalid filename"),
    ):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "Doc", upload)


# ---------------------------------------------------------------------------
# create — _write_file size exceeded mid-stream
# ---------------------------------------------------------------------------


async def test_write_file_size_exceeded_cleanup(tmp_path: Path) -> None:
    """_write_file raises InputValidationError when size limit is exceeded streaming."""
    import src.services.document as svc_mod

    repo = StubDocumentRepository()
    service = DocumentService(
        repo=repo,
        chunk_repo=StubChunkRepository(),
        session=MockSession(),  # type: ignore[arg-type]
    )
    user = _make_user()
    workspace_id = uuid.uuid4()

    # declared_size = None bypasses the early size check; MAX_UPLOAD_SIZE_MB = 0
    # means even one byte exceeds the limit inside _write_file.
    upload = _make_upload_file(size=None, content_type="text/plain", content=b"hi")

    original_dir = svc_mod.settings.UPLOAD_DIR
    original_max = svc_mod.settings.MAX_UPLOAD_SIZE_MB
    svc_mod.settings.UPLOAD_DIR = str(tmp_path)  # type: ignore[assignment]
    svc_mod.settings.MAX_UPLOAD_SIZE_MB = 0  # type: ignore[assignment]
    try:
        with pytest.raises(InputValidationError, match="exceeds maximum size"):
            await service.create(
                user, workspace_id, WorkspaceRole.MEMBER, "Doc", upload
            )
    finally:
        svc_mod.settings.UPLOAD_DIR = original_dir  # type: ignore[assignment]
        svc_mod.settings.MAX_UPLOAD_SIZE_MB = original_max  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# delete — OSError on file removal is suppressed
# ---------------------------------------------------------------------------


async def test_delete_oserror_is_suppressed() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    doc.file_path = "/some/path/file.pdf"
    repo._store[doc.id] = doc

    with patch("pathlib.Path.unlink", side_effect=OSError("disk full")):
        await service.delete(user, workspace_id, WorkspaceRole.OWNER, doc.id)

    assert doc.id not in repo._store


# ---------------------------------------------------------------------------
# get / list / get_workspace_stats — reads scoped by workspace_id
# ---------------------------------------------------------------------------


async def test_get_found_returns_doc() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    result = await service.get(workspace_id, doc.id)
    assert result.id == doc.id


async def test_get_missing_raises_not_found() -> None:
    service, _ = _make_service()
    with pytest.raises(NotFoundError):
        await service.get(uuid.uuid4(), uuid.uuid4())


async def test_list_returns_only_workspace_docs() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    other = _make_document(uuid.uuid4(), user.id)
    repo._store[doc.id] = doc
    repo._store[other.id] = other

    page = await service.list(workspace_id)
    assert [d.id for d in page.items] == [doc.id]
    assert page.has_more is False


async def test_list_hands_out_a_cursor_it_accepts_back() -> None:
    # Both adapters pass next_cursor through unchanged, so the service owns the format.
    service, repo = _make_service()
    position = Cursor(created_at=datetime(2026, 10, 22, 12, 0, 0, 1), id=uuid.uuid4())
    repo.next_cursor = position

    page = await service.list(uuid.uuid4())
    assert isinstance(page.next_cursor, str)
    assert page.has_more is True

    await service.list(uuid.uuid4(), cursor_str=page.next_cursor)
    assert repo.received_cursor == position


async def test_list_invalid_cursor_raises_validation_error() -> None:
    service, _ = _make_service()
    with pytest.raises(InputValidationError) as exc_info:
        await service.list(uuid.uuid4(), cursor_str="not-a-cursor")
    # The detail reaches MCP's model too, so it says how to recover.
    assert exc_info.value.detail == INVALID_CURSOR_DETAIL
    assert "next_cursor" in INVALID_CURSOR_DETAIL


async def test_get_workspace_stats() -> None:
    from src.domain.workspace import WorkspaceStats

    service, repo = _make_service()

    async def _stats(wid: uuid.UUID) -> WorkspaceStats:
        return WorkspaceStats(
            document_count=3,
            chunk_count=0,
            total_tokens_indexed=0,
            last_document_updated_at=None,
        )

    repo.get_workspace_stats = _stats  # type: ignore[attr-defined]

    result = await service.get_workspace_stats(uuid.uuid4())
    assert result.document_count == 3


async def test_update_nonexistent_document_raises_not_found() -> None:
    service, _ = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()

    with pytest.raises(NotFoundError):
        await service.update(
            user,
            workspace_id,
            WorkspaceRole.MEMBER,
            uuid.uuid4(),
            DocumentUpdateInput(title="X"),
        )


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def _reader(
    status: DocumentStatus = DocumentStatus.INDEXED,
) -> tuple[DocumentService, StubChunkRepository, Document]:
    repo, chunks = StubDocumentRepository(), StubChunkRepository()
    service = DocumentService(
        repo=repo,
        chunk_repo=chunks,
        session=MockSession(),  # type: ignore[arg-type]
    )
    doc = _make_document(uuid.uuid4(), uuid.uuid4())
    doc.status = status
    doc.version = 2
    repo._store[doc.id] = doc
    # Chunks 0-4 of the current version; version 1's are stale.
    chunks.add(doc, 2, 0, 1, 2, 3, 4)
    chunks.add(doc, 1, *range(9))
    return service, chunks, doc


async def test_read_returns_a_page_of_the_current_version() -> None:
    service, _, doc = _reader()

    page = await service.read(doc.workspace_id, doc.id, start_chunk=1, max_chunks=2)

    assert (page.document_id, page.title, page.total_chunks) == (doc.id, doc.title, 5)
    assert [c.chunk_text for c in page.chunks] == ["v2 1", "v2 2"]
    assert page.next_start_chunk == 3


async def test_read_last_page_has_no_next_start() -> None:
    service, _, doc = _reader()

    page = await service.read(doc.workspace_id, doc.id, start_chunk=3, max_chunks=2)

    assert [c.chunk_index for c in page.chunks] == [3, 4]
    assert page.next_start_chunk is None


async def test_read_next_start_is_the_next_stored_index() -> None:
    # Not start + page size: positions come from the stored rows.
    service, chunks, doc = _reader()
    chunks.rows = [r for r in chunks.rows if r[2].chunk_index != 3 or r[1] != 2]

    page = await service.read(doc.workspace_id, doc.id, start_chunk=1, max_chunks=2)

    assert page.next_start_chunk == 4


async def test_read_scopes_the_document_to_the_workspace() -> None:
    service, chunks, doc = _reader()

    with pytest.raises(NotFoundError):
        await service.read(uuid.uuid4(), doc.id)
    assert chunks.calls == 0


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (DocumentStatus.PENDING, NOT_INDEXED_DETAIL),
        (DocumentStatus.PROCESSING, NOT_INDEXED_DETAIL),
        (DocumentStatus.READY, NOT_INDEXED_DETAIL),
        # Retrying a failed document would loop, so it says so.
        (DocumentStatus.FAILED, FAILED_DETAIL),
    ],
)
async def test_read_refuses_a_document_that_is_not_indexed(
    status: DocumentStatus, detail: str
) -> None:
    service, chunks, doc = _reader(status)

    with pytest.raises(ConflictError) as exc_info:
        await service.read(doc.workspace_id, doc.id)
    assert exc_info.value.detail == detail
    assert chunks.calls == 0


async def test_read_past_the_end_names_the_chunk_count() -> None:
    service, _, doc = _reader()

    with pytest.raises(InputValidationError) as exc_info:
        await service.read(doc.workspace_id, doc.id, start_chunk=5)
    assert exc_info.value.detail == (
        "start_chunk 5 is past the end: the document has 5 chunks, numbered from 0"
    )
