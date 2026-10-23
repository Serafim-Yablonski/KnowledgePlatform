import codecs
import uuid
from typing import BinaryIO

import anyio
import anyio.lowlevel
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import ConflictError, InputValidationError, NotFoundError
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    DEFAULT_PAGE_SIZE,
    Cursor,
    DocumentChunks,
    DocumentPage,
    DocumentStatus,
    DocumentUpdateInput,
    DocumentUpload,
    clean_display_text,
    clean_original_filename,
    decode_cursor,
    encode_cursor,
    title_from_filename,
)
from src.domain.roles import Permission, WorkspaceRole, require_permission
from src.domain.workspace import WorkspaceStats
from src.models.document import Document
from src.models.user import User
from src.repositories.protocols import (
    ChunkRepositoryProtocol,
    DocumentRepositoryProtocol,
    FileStoreProtocol,
)

logger = structlog.get_logger(__name__)

_CHUNK = 65536

# Reaches the MCP model as well as REST clients, so it says how to recover.
INVALID_CURSOR_DETAIL = (
    "Invalid pagination cursor: pass next_cursor from the previous page unchanged, "
    "or omit it for the first page"
)
# read() refuses documents without searchable text; each message says whether waiting
# helps.
NOT_INDEXED_DETAIL = "Document is still being processed; try again shortly"
FAILED_DETAIL = "Document processing failed; it has no readable content"

_MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
}
_TEXT_TYPES = {"text/plain", "text/markdown"}

# Text uploads must be what extract_text can store: it decodes the whole file as
# strict UTF-8, and Postgres TEXT rejects NUL. Both messages reach clients.
NOT_UTF8_DETAIL = "Text files must be UTF-8 encoded; re-save the file as UTF-8"
NUL_DETAIL = (
    "Text file contains NUL bytes (a binary or UTF-16 file?); upload it as UTF-8 text"
)


def _size_error() -> InputValidationError:
    return InputValidationError(
        f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
    )


def _inspect_upload(stream: BinaryIO, content_type_str: str) -> int:
    """Check the whole stream against the size limit and its declared type.

    Runs before anything is stored, so a rejected upload leaves no row and no file.
    Returns the size and rewinds the stream for the write.
    """
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    is_text = content_type_str in _TEXT_TYPES
    # Incremental: a multibyte character split across two reads is still valid.
    decoder = codecs.getincrementaldecoder("utf-8")()
    total = 0
    chunk = stream.read(_CHUNK)
    # Checked on the first read even when it is empty: an empty PDF is no PDF.
    expected = _MAGIC_BYTES.get(content_type_str)
    if expected and not chunk.startswith(expected):
        raise InputValidationError("File content does not match declared Content-Type")
    while chunk:
        total += len(chunk)
        if total > max_bytes:
            raise _size_error()
        if is_text:
            if b"\x00" in chunk:
                raise InputValidationError(NUL_DETAIL)
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError:
                raise InputValidationError(NOT_UTF8_DETAIL) from None
        chunk = stream.read(_CHUNK)
    if is_text:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise InputValidationError(NOT_UTF8_DETAIL) from None
    stream.seek(0)
    return total


def _clean_title(raw: str) -> str:
    title = clean_display_text(raw, "Title")
    if not title:
        raise InputValidationError("Title must not be blank")
    return title


def _resolve_title(given: str | None, original_filename: str | None) -> str:
    """A given title wins; otherwise the filename's stem, as file stores do."""
    if given is not None:
        return _clean_title(given)
    if original_filename is None:
        raise InputValidationError(
            "A title is required when the file has no name: send the title field"
        )
    return title_from_filename(original_filename)


class DocumentService:
    def __init__(
        self,
        repo: DocumentRepositoryProtocol,
        chunk_repo: ChunkRepositoryProtocol,
        session: AsyncSession,
        files: FileStoreProtocol,
    ) -> None:
        self._repo = repo
        self._chunks = chunk_repo
        self._session = session
        self._files = files

    async def create(
        self,
        user: User,
        workspace_id: uuid.UUID,
        role: WorkspaceRole,
        title: str | None,
        upload: DocumentUpload,
    ) -> Document:
        require_permission(role, Permission.CREATE_DOCUMENT)
        # Reject early if the adapter already knows the size (no DB row, no disk write).
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if upload.declared_size is not None and upload.declared_size > max_bytes:
            raise _size_error()
        content_type_str = upload.content_type
        if content_type_str not in ALLOWED_CONTENT_TYPES:
            raise InputValidationError(
                f"Unsupported file type: {content_type_str!r}. "
                f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        content_type = ALLOWED_CONTENT_TYPES[content_type_str]
        original_filename = clean_original_filename(upload.filename)
        title = _resolve_title(title, original_filename)

        # The authoritative size and content check, the only one when the adapter
        # knows no size up front. Blocking reads go to a thread.
        size = await anyio.to_thread.run_sync(
            _inspect_upload, upload.stream, content_type_str
        )

        doc = await self._repo.create(
            workspace_id=workspace_id,
            title=title,
            content_type=content_type,
            file_path="",
            file_size_bytes=size,
            uploaded_by=user.id,
            original_filename=original_filename,
        )

        try:
            key = await self._files.save(
                workspace_id, doc.id, content_type, upload.stream
            )
        except OSError as exc:
            logger.error(
                "upload write failed", document_id=str(doc.id), errno=exc.errno
            )
            raise
        try:
            # The thread can't be interrupted, so a cancellation during the write is
            # held back until here: surface it before the commit point.
            await anyio.lowlevel.checkpoint()
        except BaseException:
            # Shielded: inside a cancelled scope any other await is cancelled at once.
            with anyio.CancelScope(shield=True):
                await self._files.remove(key)
            raise

        doc.file_path = key
        # The commit point. From here on the file is kept whatever happens: a failed
        # commit may still have committed, and a row without its file is corruption,
        # while a file without its row is only garbage.
        try:
            # Single commit covers both the initial flush and the file_path update.
            await self._session.commit()
        except BaseException:
            logger.error(
                "upload commit failed; file kept",
                document_id=str(doc.id),
                path=key,
            )
            raise
        # Dispatch AFTER commit so the worker can find the document record.
        from src.workers.tasks.extract_text import extract_text

        extract_text.delay(str(doc.id))
        await self._session.refresh(doc)

        return doc

    async def get(self, workspace_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        doc = await self._repo.get_by_id(workspace_id, document_id)
        if doc is None:
            raise NotFoundError("Document not found")
        return doc

    async def list(
        self,
        workspace_id: uuid.UUID,
        cursor_str: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        status: DocumentStatus | None = None,
    ) -> DocumentPage[Document]:
        try:
            cursor: Cursor | None = decode_cursor(cursor_str) if cursor_str else None
        except ValueError as exc:
            raise InputValidationError(INVALID_CURSOR_DETAIL) from exc
        docs, next_cursor = await self._repo.list_by_workspace(
            workspace_id, limit=limit, cursor=cursor, status=status
        )
        return DocumentPage(
            items=list(docs),
            next_cursor=encode_cursor(next_cursor) if next_cursor else None,
        )

    async def read(
        self,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        start_chunk: int = 0,
        max_chunks: int = 8,
    ) -> DocumentChunks:
        """Up to ``max_chunks`` of the current version's chunks from ``start_chunk``."""
        doc = await self.get(workspace_id, document_id)
        if doc.status == DocumentStatus.FAILED:
            raise ConflictError(FAILED_DETAIL)
        if doc.status != DocumentStatus.INDEXED:
            raise ConflictError(NOT_INDEXED_DETAIL)
        # One extra row tells whether more follow, and where they start.
        chunks = await self._chunks.read_page(
            doc.id, version=doc.version, start=start_chunk, limit=max_chunks + 1
        )
        total = await self._chunks.count(doc.id, version=doc.version)
        if not chunks:
            raise InputValidationError(
                f"start_chunk {start_chunk} is past the end: the document has "
                f"{total} chunks, numbered from 0"
            )
        more = len(chunks) > max_chunks
        return DocumentChunks(
            document_id=doc.id,
            title=doc.title,
            total_chunks=total,
            chunks=chunks[:max_chunks],
            next_start_chunk=chunks[max_chunks].chunk_index if more else None,
        )

    async def get_workspace_stats(self, workspace_id: uuid.UUID) -> WorkspaceStats:
        return await self._repo.get_workspace_stats(workspace_id)

    async def update(
        self,
        user: User,
        workspace_id: uuid.UUID,
        role: WorkspaceRole,
        document_id: uuid.UUID,
        data: DocumentUpdateInput,
    ) -> Document:
        require_permission(role, Permission.UPDATE_DOCUMENT)
        if data.title is not None:
            data = DocumentUpdateInput(title=_clean_title(data.title))
        doc = await self.get(workspace_id, document_id)
        doc = await self._repo.update(doc, data)
        await self._session.commit()
        return doc

    async def delete(
        self,
        user: User,
        workspace_id: uuid.UUID,
        role: WorkspaceRole,
        document_id: uuid.UUID,
    ) -> None:
        require_permission(role, Permission.DELETE_DOCUMENT)
        doc = await self.get(workspace_id, document_id)
        file_path = doc.file_path
        # Delete the DB record first — a consistent DB with an orphaned file is better
        # than a deleted file with a stale DB row pointing at nothing.
        await self._repo.delete(doc)
        await self._session.commit()
        if file_path:
            try:
                await self._files.remove(file_path)
            except OSError as exc:
                logger.warning(
                    "failed to delete document file",
                    path=file_path,
                    error=str(exc),
                )
