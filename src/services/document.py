import uuid
from pathlib import Path
from typing import IO

import anyio
import structlog
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import ForbiddenError, InputValidationError, NotFoundError
from src.domain.documents import (
    ALLOWED_CONTENT_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
    Cursor,
    DocumentStatus,
    decode_cursor,
    encode_cursor,
)
from src.domain.roles import PERMISSIONS, WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.protocols import DocumentRepositoryProtocol
from src.schemas.document import DocumentResponse, DocumentUpdate, PaginatedResponse

logger = structlog.get_logger(__name__)

_CHUNK = 65536


def _require_permission(role: WorkspaceRole, permission: str) -> None:
    if permission not in PERMISSIONS[role]:
        raise ForbiddenError("Insufficient permissions")


def _write_file(src: IO[bytes], dest_path: Path) -> int:
    """Stream src to dest_path in chunks, enforcing MAX_UPLOAD_SIZE_BYTES."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest_path.open("wb") as dest:
        while chunk := src.read(_CHUNK):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE_BYTES:
                dest_path.unlink(missing_ok=True)
                raise InputValidationError(
                    f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
                )
            dest.write(chunk)
    return total


class DocumentService:
    def __init__(self, repo: DocumentRepositoryProtocol, session: AsyncSession) -> None:
        self._repo = repo
        self._session = session

    async def create(
        self,
        user: User,
        workspace: Workspace,
        role: WorkspaceRole,
        title: str,
        file: UploadFile,
    ) -> DocumentResponse:
        _require_permission(role, "create_document")
        # Fast-fail if Content-Length header already indicates oversized payload.
        if file.size is not None and file.size > MAX_UPLOAD_SIZE_BYTES:
            raise InputValidationError(
                f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB"
            )
        content_type_str = file.content_type or ""
        if content_type_str not in ALLOWED_CONTENT_TYPES:
            raise InputValidationError(
                f"Unsupported file type: {content_type_str!r}. "
                f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        content_type = ALLOWED_CONTENT_TYPES[content_type_str]

        # Flush to get the server-generated ID, then build the real path.
        doc = await self._repo.create(
            workspace_id=workspace.id,
            title=title,
            content_type=content_type,
            file_path="",
            file_size_bytes=file.size or 0,
            uploaded_by=user.id,
        )

        # Basename-only filename prevents path traversal via "../../" sequences.
        filename = Path(file.filename or "upload").name or "upload"
        file_path = (
            Path(settings.UPLOAD_DIR) / str(workspace.id) / str(doc.id) / filename
        )
        upload_root = Path(settings.UPLOAD_DIR).resolve()
        if not file_path.resolve().is_relative_to(upload_root):
            raise InputValidationError("Invalid filename")

        # Offload blocking I/O to a thread; _write_file also enforces the size limit
        # for streaming uploads where Content-Length is absent.
        actual_size = await anyio.to_thread.run_sync(
            lambda: _write_file(file.file, file_path)
        )

        doc.file_path = str(file_path)
        doc.file_size_bytes = actual_size
        # Single commit covers both the initial flush and the file_path update.
        await self._session.commit()
        await self._session.refresh(doc)

        return DocumentResponse.model_validate(doc)

    async def get(
        self, user: User, workspace: Workspace, document_id: uuid.UUID
    ) -> DocumentResponse:
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace.id:
            raise NotFoundError("Document not found")
        return DocumentResponse.model_validate(doc)

    async def list(
        self,
        user: User,
        workspace: Workspace,
        cursor_str: str | None,
        limit: int,
        status: DocumentStatus | None = None,
    ) -> PaginatedResponse[DocumentResponse]:
        cursor: Cursor | None = decode_cursor(cursor_str) if cursor_str else None
        docs, next_cursor = await self._repo.list_by_workspace(
            workspace.id, limit=limit, cursor=cursor, status=status
        )
        return PaginatedResponse(
            items=[DocumentResponse.model_validate(d) for d in docs],
            next_cursor=encode_cursor(next_cursor) if next_cursor else None,
            has_more=next_cursor is not None,
        )

    async def update(
        self,
        user: User,
        workspace: Workspace,
        role: WorkspaceRole,
        document_id: uuid.UUID,
        data: DocumentUpdate,
    ) -> DocumentResponse:
        _require_permission(role, "update_document")
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.workspace_id != workspace.id:
            raise NotFoundError("Document not found")
        doc = await self._repo.update(doc, data)
        return DocumentResponse.model_validate(doc)

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
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "failed to delete document file",
                    path=file_path,
                    error=str(exc),
                )
