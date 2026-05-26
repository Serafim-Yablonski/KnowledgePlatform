import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import anyio
import structlog
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.cache import ResponseCache
from src.core.config import settings
from src.core.exceptions import ForbiddenError, InputValidationError, NotFoundError
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    Cursor,
    DocumentPage,
    DocumentStatus,
    DocumentUpdateInput,
    decode_cursor,
)
from src.domain.roles import PERMISSIONS, WorkspaceRole
from src.domain.workspace import WorkspaceStats
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.protocols import DocumentRepositoryProtocol

logger = structlog.get_logger(__name__)

_CHUNK = 65536

_MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
}
_TEXT_TYPES = {"text/plain", "text/markdown"}


async def _validate_magic_bytes(file: UploadFile, content_type_str: str) -> None:
    header = await file.read(8)
    await file.seek(0)
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


def _require_permission(role: WorkspaceRole, permission: str) -> None:
    if permission not in PERMISSIONS[role]:
        raise ForbiddenError("Insufficient permissions")


def _write_file(src: IO[bytes], dest_path: Path) -> int:
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
        cache: ResponseCache | None = None,
    ) -> None:
        self._repo = repo
        self._session = session
        self._cache = cache

    async def create(
        self,
        user: User,
        workspace: Workspace,
        role: WorkspaceRole,
        title: str,
        file: UploadFile,
    ) -> Document:
        _require_permission(role, "create_document")
        # Fast-fail if Content-Length header already indicates oversized payload.
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file.size is not None and file.size > max_bytes:
            raise InputValidationError(
                f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
            )
        content_type_str = file.content_type or ""
        if content_type_str not in ALLOWED_CONTENT_TYPES:
            raise InputValidationError(
                f"Unsupported file type: {content_type_str!r}. "
                f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        await _validate_magic_bytes(file, content_type_str)
        content_type = ALLOWED_CONTENT_TYPES[content_type_str]

        filename = Path(file.filename or "upload").name or "upload"
        _probe = (
            Path(settings.UPLOAD_DIR)
            / "00000000-0000-0000-0000-000000000000"
            / "00000000-0000-0000-0000-000000000000"
            / filename
        )
        if not _probe.resolve().is_relative_to(Path(settings.UPLOAD_DIR).resolve()):
            raise InputValidationError("Invalid filename")

        doc = await self._repo.create(
            workspace_id=workspace.id,
            title=title,
            content_type=content_type,
            file_path="",
            file_size_bytes=file.size or 0,
            uploaded_by=user.id,
        )

        file_path = (
            Path(settings.UPLOAD_DIR) / str(workspace.id) / str(doc.id) / filename
        )

        # Offload blocking I/O to a thread; _write_file also enforces the size limit
        # for streaming uploads where Content-Length is absent.
        actual_size = await anyio.to_thread.run_sync(
            lambda: _write_file(file.file, file_path)
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

    async def _get_by_id(
        self,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> Document:
        """Fetch a document scoped to a workspace. Caller must verify membership."""
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace_id:
            raise NotFoundError("Document not found")
        return doc

    async def get(
        self, user: User, workspace: Workspace, document_id: uuid.UUID
    ) -> Document:
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace.id:
            raise NotFoundError("Document not found")
        return doc

    async def list(
        self,
        user: User,
        workspace: Workspace,
        cursor_str: str | None,
        limit: int,
        status: DocumentStatus | None = None,
    ) -> DocumentPage[Document]:
        try:
            cursor: Cursor | None = decode_cursor(cursor_str) if cursor_str else None
        except ValueError as exc:
            raise InputValidationError("Invalid pagination cursor") from exc
        docs, next_cursor = await self._repo.list_by_workspace(
            workspace.id, limit=limit, cursor=cursor, status=status
        )
        return DocumentPage(
            items=list(docs),
            next_cursor=next_cursor,
            has_more=next_cursor is not None,
        )

    async def _list_by_workspace_id(
        self,
        workspace_id: uuid.UUID,
        limit: int = 50,
    ) -> Sequence[Document]:
        """List documents by workspace ID. Caller must verify workspace membership."""
        docs, _ = await self._repo.list_by_workspace(workspace_id, limit=limit)
        return docs

    async def _get_workspace_stats(self, workspace_id: uuid.UUID) -> WorkspaceStats:
        """Return aggregate stats for a workspace. Caller must verify membership."""
        return await self._repo.get_workspace_stats(workspace_id)

    async def update(
        self,
        user: User,
        workspace: Workspace,
        role: WorkspaceRole,
        document_id: uuid.UUID,
        data: DocumentUpdateInput,
    ) -> Document:
        _require_permission(role, "update_document")
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace.id:
            raise NotFoundError("Document not found")
        doc = await self._repo.update(doc, data)
        await self._session.commit()
        if self._cache is not None:
            await self._cache.delete_pattern(f"search:{workspace.id}:*")
        return doc

    async def delete(
        self,
        user: User,
        workspace: Workspace,
        role: WorkspaceRole,
        document_id: uuid.UUID,
    ) -> None:
        _require_permission(role, "delete_document")
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace.id:
            raise NotFoundError("Document not found")
        file_path = doc.file_path
        # Delete the DB record first — a consistent DB with an orphaned file is better
        # than a deleted file with a stale DB row pointing at nothing.
        await self._repo.delete(doc)
        await self._session.commit()
        if self._cache is not None:
            await self._cache.delete_pattern(f"search:{workspace.id}:*")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "failed to delete document file",
                    path=file_path,
                    error=str(exc),
                )
