import uuid
from pathlib import Path
from typing import BinaryIO

import anyio
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import InputValidationError, NotFoundError
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    Cursor,
    DocumentPage,
    DocumentStatus,
    DocumentUpdateInput,
    DocumentUpload,
    decode_cursor,
)
from src.domain.roles import Permission, WorkspaceRole, require_permission
from src.domain.workspace import WorkspaceStats
from src.models.document import Document
from src.models.user import User
from src.repositories.protocols import DocumentRepositoryProtocol

logger = structlog.get_logger(__name__)

_CHUNK = 65536

_MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
}
_TEXT_TYPES = {"text/plain", "text/markdown"}


def _read_header(stream: BinaryIO) -> bytes:
    header = stream.read(8)
    stream.seek(0)
    return header


async def _validate_magic_bytes(stream: BinaryIO, content_type_str: str) -> None:
    header = await anyio.to_thread.run_sync(_read_header, stream)
    expected = _MAGIC_BYTES.get(content_type_str)
    if expected and not header.startswith(expected):
        raise InputValidationError("File content does not match declared Content-Type")
    if content_type_str in _TEXT_TYPES:
        try:
            header.decode("utf-8")
        except UnicodeDecodeError:
            raise InputValidationError(
                "File content does not match declared Content-Type"
            ) from None


def _write_file(src: BinaryIO, dest_path: Path) -> int:
    """Stream src to dest_path in chunks, enforcing the configured upload size limit."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest_path.open("wb") as dest:
        while chunk := src.read(_CHUNK):
            total += len(chunk)
            if total > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                dest_path.unlink(missing_ok=True)
                raise InputValidationError(
                    f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
                )
            dest.write(chunk)
    return total


class DocumentService:
    def __init__(
        self,
        repo: DocumentRepositoryProtocol,
        session: AsyncSession,
    ) -> None:
        self._repo = repo
        self._session = session

    async def create(
        self,
        user: User,
        workspace_id: uuid.UUID,
        role: WorkspaceRole,
        title: str,
        upload: DocumentUpload,
    ) -> Document:
        require_permission(role, Permission.CREATE_DOCUMENT)
        # Reject early if the adapter already knows the size (no DB row, no disk write).
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if upload.declared_size is not None and upload.declared_size > max_bytes:
            raise InputValidationError(
                f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
            )
        content_type_str = upload.content_type
        if content_type_str not in ALLOWED_CONTENT_TYPES:
            raise InputValidationError(
                f"Unsupported file type: {content_type_str!r}. "
                f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        await _validate_magic_bytes(upload.stream, content_type_str)
        content_type = ALLOWED_CONTENT_TYPES[content_type_str]

        filename = Path(upload.filename or "upload").name or "upload"
        _probe = (
            Path(settings.UPLOAD_DIR)
            / "00000000-0000-0000-0000-000000000000"
            / "00000000-0000-0000-0000-000000000000"
            / filename
        )
        if not _probe.resolve().is_relative_to(Path(settings.UPLOAD_DIR).resolve()):
            raise InputValidationError("Invalid filename")

        doc = await self._repo.create(
            workspace_id=workspace_id,
            title=title,
            content_type=content_type,
            file_path="",
            file_size_bytes=upload.declared_size or 0,
            uploaded_by=user.id,
        )

        file_path = (
            Path(settings.UPLOAD_DIR) / str(workspace_id) / str(doc.id) / filename
        )

        # Offload blocking I/O to a thread; _write_file enforces the authoritative size
        # limit, which is the only one when the adapter knows no size up front.
        actual_size = await anyio.to_thread.run_sync(
            lambda: _write_file(upload.stream, file_path)
        )

        doc.file_path = str(file_path)
        doc.file_size_bytes = actual_size
        # Single commit covers both the initial flush and the file_path update.
        await self._session.commit()
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
        limit: int = 20,
        status: DocumentStatus | None = None,
    ) -> DocumentPage[Document]:
        try:
            cursor: Cursor | None = decode_cursor(cursor_str) if cursor_str else None
        except ValueError as exc:
            raise InputValidationError("Invalid pagination cursor") from exc
        docs, next_cursor = await self._repo.list_by_workspace(
            workspace_id, limit=limit, cursor=cursor, status=status
        )
        return DocumentPage(
            items=list(docs),
            next_cursor=next_cursor,
            has_more=next_cursor is not None,
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
                Path(file_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "failed to delete document file",
                    path=file_path,
                    error=str(exc),
                )
