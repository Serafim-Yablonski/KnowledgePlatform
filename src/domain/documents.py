import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    INDEXED = "indexed"  # chunks stored and searchable
    FAILED = "failed"


class ContentType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    PLAINTEXT = "plaintext"


ALLOWED_CONTENT_TYPES: dict[str, ContentType] = {
    "application/pdf": ContentType.PDF,
    "text/markdown": ContentType.MARKDOWN,
    "text/plain": ContentType.PLAINTEXT,
}


# One page of a document listing, the same for REST and MCP. The service trusts it:
# the repository fetches limit + 1 rows, so 0 would crash.
DocumentPageSize = Annotated[int, Field(ge=1, le=100)]
DEFAULT_PAGE_SIZE = 20


@dataclass
class Cursor:
    created_at: datetime
    id: uuid.UUID


def encode_cursor(cursor: Cursor) -> str:
    payload = json.dumps(
        {"created_at": cursor.created_at.isoformat(), "id": str(cursor.id)}
    )
    return base64.urlsafe_b64encode(payload.encode()).decode()


class _CursorPayload(BaseModel):
    # Exactly what encode_cursor writes. Naive because created_at is a naive column:
    # asyncpg refuses an aware value for it, so an aware cursor was forged.
    model_config = ConfigDict(extra="forbid")

    created_at: NaiveDatetime
    id: uuid.UUID


def decode_cursor(value: str) -> Cursor:
    if len(value) > 512:
        raise ValueError("Invalid pagination cursor")
    try:
        # pydantic's ValidationError and base64's binascii.Error are ValueErrors.
        data = _CursorPayload.model_validate_json(base64.urlsafe_b64decode(value))
    except ValueError as exc:
        raise ValueError("Invalid pagination cursor") from exc
    return Cursor(created_at=data.created_at, id=data.id)


@dataclass(frozen=True)
class DocumentUpload:
    """An uploaded file as the service sees it, independent of the web framework.

    ``declared_size`` is whatever size the adapter already knows, if any. From REST it
    is Starlette's count of the already-buffered body, not a header, so rejecting on it
    saves the DB row and disk write, not the transfer. The limit that counts is the
    one enforced while streaming ``stream`` to disk.
    """

    filename: str
    content_type: str
    declared_size: int | None
    stream: BinaryIO


@dataclass
class DocumentUpdateInput:
    title: str | None = None


@dataclass(frozen=True)
class ChunkText:
    chunk_index: int  # position in the document's current version, from 0
    section: str | None  # markdown heading path; None for other formats
    chunk_text: str


@dataclass(frozen=True)
class DocumentChunks:
    """A run of a document's chunks, in order, and where the next run starts."""

    document_id: uuid.UUID
    title: str
    total_chunks: int
    chunks: list[ChunkText]
    next_start_chunk: int | None  # None after the last chunk


@dataclass
class DocumentPage[T]:
    items: list[T]
    # Opaque: produced by encode_cursor, accepted back by the same service.
    next_cursor: str | None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None
