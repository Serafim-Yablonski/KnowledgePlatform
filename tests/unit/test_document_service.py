"""Unit tests for DocumentService using in-memory Protocol stubs."""

import errno
import io
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import logfire
import pytest
import structlog

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
    FailureReason,
)
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.repositories.file_store import LocalFileStore
from src.services.document import (
    _CHUNK,
    FAILED_DETAIL,
    FAILED_DETAILS,
    INVALID_CURSOR_DETAIL,
    NOT_INDEXED_DETAIL,
    DocumentService,
)
from tests.support.unit_of_work import FakeUnitOfWork

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
        original_filename: str | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> Document:
        doc = Document()
        doc.id = uuid.uuid4()
        doc.trace_context = trace_context
        doc.workspace_id = workspace_id
        doc.title = title
        doc.original_filename = original_filename
        doc.content_type = content_type
        doc.raw_text = None
        doc.file_path = file_path
        doc.file_size_bytes = file_size_bytes
        doc.uploaded_by = uploaded_by
        doc.status = DocumentStatus.PENDING
        doc.created_at = datetime.now(UTC).replace(tzinfo=None)
        doc.updated_at = datetime.now(UTC).replace(tzinfo=None)
        self._store[doc.id] = doc
        return doc

    async def set_file_path(self, document: Document, file_path: str) -> None:
        document.file_path = file_path

    async def get_by_id(
        self, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> Document | None:
        doc = self._store.get(document_id)
        return doc if doc is not None and doc.workspace_id == workspace_id else None

    async def update(self, document: Document, data: DocumentUpdateInput) -> Document:
        if data.title is not None:
            document.title = data.title
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
    """In memory, filtered like the real query: document, start, order."""

    def __init__(self) -> None:
        self.rows: list[tuple[uuid.UUID, ChunkText]] = []
        self.calls = 0

    def add(self, document_id: uuid.UUID, *indexes: int) -> None:
        for i in indexes:
            text = ChunkText(chunk_index=i, section=None, chunk_text=f"chunk {i}")
            self.rows.append((document_id, text))

    def _of(self, document_id: uuid.UUID) -> list[ChunkText]:
        return sorted(
            (c for d, c in self.rows if d == document_id),
            key=lambda c: c.chunk_index,
        )

    async def read_page(
        self, document_id: uuid.UUID, *, start: int, limit: int
    ) -> list[ChunkText]:
        self.calls += 1
        chunks = self._of(document_id)
        return [c for c in chunks if c.chunk_index >= start][:limit]

    async def count(self, document_id: uuid.UUID) -> int:
        self.calls += 1
        return len(self._of(document_id))


# Tests that store files pass a temp root; elsewhere a write fails loudly.
_NO_UPLOADS = Path("/nonexistent-upload-root")


def _make_service(
    root: Path = _NO_UPLOADS,
) -> tuple[DocumentService, StubDocumentRepository]:
    repo = StubDocumentRepository()
    uow = FakeUnitOfWork(documents=repo, chunks=StubChunkRepository())
    service = DocumentService(uow, files=LocalFileStore(root))
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
        service, _ = _make_service(tmp_path)
        upload = _make_upload_file(content_type=mime, filename="file.txt")
        result = await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Title", upload
        )
        assert result.content_type == expected_ct


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


async def test_update_sets_the_title() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()

    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    result = await service.update(
        user,
        workspace_id,
        WorkspaceRole.MEMBER,
        doc.id,
        DocumentUpdateInput(title="New"),
    )
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

    with pytest.raises(InputValidationError, match="UTF-8"):
        await service.create(user, workspace_id, WorkspaceRole.MEMBER, "Doc", upload)


# ---------------------------------------------------------------------------
# create — content pre-pass: whole stream checked before anything is stored
# ---------------------------------------------------------------------------


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    return tmp_path


def _text_upload(content: bytes) -> DocumentUpload:
    # declared_size=None: only the pre-pass sees the content and its size.
    return _make_upload_file(
        size=None, content_type="text/plain", filename="a.txt", content=content
    )


@pytest.mark.parametrize(
    "content",
    [
        # "é" is two bytes; the old check decoded only the first 8 and split it.
        pytest.param("abcdefgé".encode(), id="multibyte-across-byte-8"),
        pytest.param(
            b"a" * (_CHUNK - 1) + "é".encode() + b"tail", id="multibyte-across-a-read"
        ),
    ],
)
async def test_valid_utf8_split_across_reads_is_accepted(
    upload_dir: Path, content: bytes
) -> None:
    service, _ = _make_service(upload_dir)
    doc = await service.create(
        _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", _text_upload(content)
    )
    assert Path(doc.file_path).read_bytes() == content


@pytest.mark.parametrize(
    ("content", "match"),
    [
        pytest.param(b"valid header \xff later", "UTF-8", id="invalid-after-header"),
        pytest.param(b"ends mid-character \xc3", "UTF-8", id="truncated-at-eof"),
        pytest.param("hi".encode("utf-16-le"), "NUL", id="utf16-without-bom"),
    ],
)
async def test_text_that_the_worker_cannot_store_is_rejected(
    upload_dir: Path, content: bytes, match: str
) -> None:
    service, _ = _make_service(upload_dir)
    with pytest.raises(InputValidationError, match=match):
        await service.create(
            _make_user(),
            uuid.uuid4(),
            WorkspaceRole.MEMBER,
            "Doc",
            _text_upload(content),
        )


@pytest.mark.parametrize(
    ("content_type", "content", "max_mb"),
    [
        pytest.param("text/plain", b"hi", 0, id="too-large"),
        pytest.param("application/pdf", b"not a pdf", 50, id="pdf-magic"),
        pytest.param("application/pdf", b"", 50, id="empty-pdf"),
        pytest.param("text/plain", b"ok \xff", 50, id="invalid-utf8"),
        pytest.param("text/plain", b"a\x00b", 50, id="nul"),
    ],
)
async def test_rejected_upload_stores_nothing(
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    content: bytes,
    max_mb: int,
) -> None:
    import src.services.document as svc_mod

    monkeypatch.setattr(svc_mod.settings, "MAX_UPLOAD_SIZE_MB", max_mb)
    service, repo = _make_service(upload_dir)
    upload = _make_upload_file(size=None, content_type=content_type, content=content)

    with pytest.raises(InputValidationError):
        await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", upload
        )

    assert repo._store == {}
    assert list(upload_dir.iterdir()) == []


async def test_size_comes_from_the_stream_when_none_is_declared(
    upload_dir: Path,
) -> None:
    service, _ = _make_service(upload_dir)
    doc = await service.create(
        _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", _text_upload(b"12345")
    )
    assert doc.file_size_bytes == 5


# ---------------------------------------------------------------------------
# create — storage path is chosen by the server, never from the client's filename
# ---------------------------------------------------------------------------

_EXPECTED_EXTENSION = {
    ContentType.PDF: ".pdf",
    ContentType.MARKDOWN: ".md",
    ContentType.PLAINTEXT: ".txt",
}


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("..", id="dotdot"),
        pytest.param("../../etc/passwd", id="traversal"),
        pytest.param("", id="empty"),
        pytest.param("C:\\x\\a.txt", id="windows-path"),
    ],
)
async def test_storage_path_never_uses_the_client_filename(
    upload_dir: Path, filename: str
) -> None:
    service, _ = _make_service(upload_dir)
    workspace_id = uuid.uuid4()
    upload = _make_upload_file(
        content_type="text/plain", filename=filename, content=b"body"
    )

    doc = await service.create(
        _make_user(), workspace_id, WorkspaceRole.MEMBER, "Doc", upload
    )

    assert Path(doc.file_path) == upload_dir / str(workspace_id) / f"{doc.id}.txt"
    assert Path(doc.file_path).read_bytes() == b"body"


@pytest.mark.parametrize("mime", list(ALLOWED_CONTENT_TYPES))
async def test_stored_extension_follows_the_content_type(
    upload_dir: Path, mime: str
) -> None:
    service, _ = _make_service(upload_dir)
    upload = _make_upload_file(content_type=mime, filename="report.exe")

    doc = await service.create(
        _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", upload
    )

    expected = _EXPECTED_EXTENSION[ALLOWED_CONTENT_TYPES[mime]]
    assert Path(doc.file_path).suffix == expected


async def test_delete_leaves_nothing_in_the_workspace_directory(
    upload_dir: Path,
) -> None:
    service, _ = _make_service(upload_dir)
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = await service.create(
        user, workspace_id, WorkspaceRole.MEMBER, "Doc", _text_upload(b"body")
    )

    await service.delete(user, workspace_id, WorkspaceRole.OWNER, doc.id)

    assert list((upload_dir / str(workspace_id)).iterdir()) == []


async def test_existing_file_at_the_target_is_never_overwritten(
    upload_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, repo = _make_service(upload_dir)
    workspace_id = uuid.uuid4()
    create = repo.create
    occupied: list[Path] = []

    async def create_onto_an_occupied_path(**kwargs: Any) -> Document:
        # As if a restored volume already held a file under the new document's name.
        doc = await create(**kwargs)
        target = upload_dir / str(workspace_id) / f"{doc.id}.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"someone else's document")
        occupied.append(target)
        return doc

    monkeypatch.setattr(repo, "create", create_onto_an_occupied_path)

    upload = _text_upload(b"new")
    with pytest.raises(FileExistsError):
        await service.create(
            _make_user(), workspace_id, WorkspaceRole.MEMBER, "Doc", upload
        )

    assert occupied[0].read_bytes() == b"someone else's document"


async def test_an_upload_stores_its_trace_context_for_the_workers(
    upload_dir: Path,
) -> None:
    """The workers continue the upload's trace (ADR-027): the row carries the
    context of the span the upload ran in."""
    service, _ = _make_service(upload_dir)
    with logfire.span("upload") as span:
        doc = await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", _text_upload(b"x")
        )
        ids = span.get_span_context()

    assert doc.trace_context is not None
    _, trace_id, span_id, _ = doc.trace_context["traceparent"].split("-")
    assert (trace_id, span_id) == (f"{ids.trace_id:032x}", f"{ids.span_id:016x}")


# ---------------------------------------------------------------------------
# create/update — original filename and title
# ---------------------------------------------------------------------------


async def _create_named(
    root: Path, title: str | None, filename: str
) -> tuple[Document, StubDocumentRepository]:
    service, repo = _make_service(root)
    upload = _make_upload_file(content_type="text/plain", filename=filename)
    doc = await service.create(
        _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, title, upload
    )
    return doc, repo


async def test_title_defaults_to_the_filename_stem(upload_dir: Path) -> None:
    doc, _ = await _create_named(upload_dir, None, "report-v3.pdf")
    assert (doc.title, doc.original_filename) == ("report-v3", "report-v3.pdf")


async def test_given_title_and_filename_are_both_kept(upload_dir: Path) -> None:
    doc, _ = await _create_named(upload_dir, " Quarterly report ", "C:\\tmp\\q3.txt")
    assert (doc.title, doc.original_filename) == ("Quarterly report", "q3.txt")


async def test_title_required_when_the_file_has_no_name(upload_dir: Path) -> None:
    with pytest.raises(InputValidationError, match="title is required"):
        await _create_named(upload_dir, None, "")


@pytest.mark.parametrize(
    ("title", "filename", "match"),
    [
        pytest.param("Doc", "invoice\u202efdp.exe", "Filename contains", id="filename"),
        pytest.param("a\nb", "a.txt", "Title contains", id="title-control"),
        pytest.param("   ", "a.txt", "Title must not be blank", id="title-blank"),
    ],
)
async def test_unsafe_names_are_rejected_before_anything_is_stored(
    upload_dir: Path, title: str, filename: str, match: str
) -> None:
    service, repo = _make_service(upload_dir)
    upload = _make_upload_file(content_type="text/plain", filename=filename)

    with pytest.raises(InputValidationError, match=match):
        await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, title, upload
        )

    assert repo._store == {}
    assert list(upload_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("title", "match"),
    [
        pytest.param("a\u202eb", "Title contains", id="bidi"),
        pytest.param(" ", "Title must not be blank", id="blank"),
    ],
)
async def test_update_rejects_unsafe_titles(title: str, match: str) -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    with pytest.raises(InputValidationError, match=match):
        await service.update(
            user,
            workspace_id,
            WorkspaceRole.MEMBER,
            doc.id,
            DocumentUpdateInput(title=title),
        )
    assert doc.title == "Test Doc"


async def test_update_stores_the_cleaned_title() -> None:
    service, repo = _make_service()
    user = _make_user()
    workspace_id = uuid.uuid4()
    doc = _make_document(workspace_id, user.id)
    repo._store[doc.id] = doc

    result = await service.update(
        user,
        workspace_id,
        WorkspaceRole.MEMBER,
        doc.id,
        DocumentUpdateInput(title=" New "),
    )
    assert result.title == "New"


# ---------------------------------------------------------------------------
# create — after the row: failed writes leave no file; a 201 survives a crash
# ---------------------------------------------------------------------------


class _FailsWhileWriting(io.BytesIO):
    """Passes the pre-pass, then fails on the write pass after one chunk is written."""

    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._rewinds = 0
        self._reads_after_rewind = 0

    def seek(self, pos: int, whence: int = 0) -> int:
        self._rewinds += 1
        return super().seek(pos, whence)

    def read(self, size: int | None = -1) -> bytes:
        if self._rewinds:
            self._reads_after_rewind += 1
            if self._reads_after_rewind == 2:
                raise OSError(errno.EIO, "I/O error")
        return super().read(size)


def _workspace_files(upload_dir: Path) -> list[Path]:
    return [p for p in upload_dir.rglob("*") if p.is_file()]


async def test_write_error_leaves_no_file_and_logs_errno(upload_dir: Path) -> None:
    service, _ = _make_service(upload_dir)
    upload = DocumentUpload(
        filename="a.txt",
        content_type="text/plain",
        declared_size=None,
        stream=_FailsWhileWriting(b"a" * (_CHUNK + 10)),
    )

    with structlog.testing.capture_logs() as logs, pytest.raises(OSError):
        await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", upload
        )

    assert _workspace_files(upload_dir) == []
    assert any(
        e["event"] == "upload write failed" and e["errno"] == errno.EIO for e in logs
    )


async def test_cancelled_during_the_write_leaves_no_file(
    upload_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.repositories.file_store as store_mod

    write = store_mod._write_file

    def slow_write(src: Any, dest: Path) -> None:
        time.sleep(0.2)
        write(src, dest)

    monkeypatch.setattr(store_mod, "_write_file", slow_write)
    service, _ = _make_service(upload_dir)

    # The thread finishes the write; the cancellation must surface before commit.
    with anyio.move_on_after(0.05) as scope:
        await service.create(
            _make_user(), uuid.uuid4(), WorkspaceRole.MEMBER, "Doc", _text_upload(b"x")
        )

    assert scope.cancelled_caught
    assert _workspace_files(upload_dir) == []


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
    failure_reason: FailureReason | None = None,
) -> tuple[DocumentService, StubChunkRepository, Document]:
    repo, chunks = StubDocumentRepository(), StubChunkRepository()
    uow = FakeUnitOfWork(documents=repo, chunks=chunks)
    service = DocumentService(uow, files=LocalFileStore(_NO_UPLOADS))
    doc = _make_document(uuid.uuid4(), uuid.uuid4())
    doc.status = status
    doc.failure_reason = failure_reason
    repo._store[doc.id] = doc
    # Chunks 0-4; another document's are never read.
    chunks.add(doc.id, 0, 1, 2, 3, 4)
    chunks.add(uuid.uuid4(), *range(9))
    return service, chunks, doc


async def test_read_returns_a_page_of_the_documents_chunks() -> None:
    service, _, doc = _reader()

    page = await service.read(doc.workspace_id, doc.id, start_chunk=1, max_chunks=2)

    assert (page.document_id, page.title, page.total_chunks) == (doc.id, doc.title, 5)
    assert [c.chunk_text for c in page.chunks] == ["chunk 1", "chunk 2"]
    assert page.next_start_chunk == 3


async def test_read_last_page_has_no_next_start() -> None:
    service, _, doc = _reader()

    page = await service.read(doc.workspace_id, doc.id, start_chunk=3, max_chunks=2)

    assert [c.chunk_index for c in page.chunks] == [3, 4]
    assert page.next_start_chunk is None


async def test_read_next_start_is_the_next_stored_index() -> None:
    # Not start + page size: positions come from the stored rows.
    service, chunks, doc = _reader()
    chunks.rows = [r for r in chunks.rows if r[1].chunk_index != 3 or r[0] != doc.id]

    page = await service.read(doc.workspace_id, doc.id, start_chunk=1, max_chunks=2)

    assert page.next_start_chunk == 4


async def test_read_scopes_the_document_to_the_workspace() -> None:
    service, chunks, doc = _reader()

    with pytest.raises(NotFoundError):
        await service.read(uuid.uuid4(), doc.id)
    assert chunks.calls == 0


@pytest.mark.parametrize(
    ("status", "reason", "detail"),
    [
        (DocumentStatus.PENDING, None, NOT_INDEXED_DETAIL),
        (DocumentStatus.PROCESSING, None, NOT_INDEXED_DETAIL),
        (DocumentStatus.READY, None, NOT_INDEXED_DETAIL),
        # Retrying a failed document would loop, so it says so, and why when known.
        (DocumentStatus.FAILED, None, FAILED_DETAIL),
        *[
            (DocumentStatus.FAILED, reason, FAILED_DETAILS[reason])
            for reason in FailureReason
        ],
    ],
)
async def test_read_refuses_a_document_that_is_not_indexed(
    status: DocumentStatus, reason: FailureReason | None, detail: str
) -> None:
    service, chunks, doc = _reader(status, reason)

    with pytest.raises(ConflictError) as exc_info:
        await service.read(doc.workspace_id, doc.id)
    assert exc_info.value.detail == detail
    assert chunks.calls == 0


def test_every_failure_reason_has_its_own_detail() -> None:
    # The model reads the detail to decide what to tell the user; a reason that fell
    # back to the generic text, or shared another's, would hide why it failed.
    assert set(FAILED_DETAILS) == set(FailureReason)
    assert len(set(FAILED_DETAILS.values())) == len(FailureReason)
    assert FAILED_DETAIL not in FAILED_DETAILS.values()


async def test_read_past_the_end_names_the_chunk_count() -> None:
    service, _, doc = _reader()

    with pytest.raises(InputValidationError) as exc_info:
        await service.read(doc.workspace_id, doc.id, start_chunk=5)
    assert exc_info.value.detail == (
        "start_chunk 5 is past the end: the document has 5 chunks, numbered from 0"
    )
