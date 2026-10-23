import base64
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePath
from typing import Annotated, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime

from src.core.exceptions import InputValidationError


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


# Titles and original filenames are shown to people, sent in headers and put into
# prompts. Bidi overrides/isolates reorder what is displayed ("invoice\u202efdp.exe"
# reads as "invoiceexe.pdf"). Zero-width joiners and LRM/RLM stay: real names use them.
_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
MAX_DISPLAY_TEXT_LENGTH = 255  # also the longest filename any common filesystem allows


def clean_display_text(raw: str, what: str) -> str:
    """NFC and trimmed; rejects control and bidi-override characters and overlong text.

    ``what`` names the field in the error, which reaches clients.
    """
    text = unicodedata.normalize("NFC", raw).strip()
    for char in text:
        if unicodedata.category(char) == "Cc" or char in _BIDI_CONTROLS:
            raise InputValidationError(
                f"{what} contains the invisible character U+{ord(char):04X}"
            )
    if len(text) > MAX_DISPLAY_TEXT_LENGTH:
        raise InputValidationError(
            f"{what} is longer than {MAX_DISPLAY_TEXT_LENGTH} characters"
        )
    return text


def clean_original_filename(raw: str) -> str | None:
    """The last path segment as display text; None when there is no name."""
    name = clean_display_text(re.split(r"[/\\]", raw)[-1], "Filename")
    return None if name in ("", ".", "..") else name


def title_from_filename(name: str) -> str:
    return PurePath(name).stem or name


@dataclass(frozen=True)
class DocumentUpload:
    """An uploaded file as the service sees it, independent of the web framework.

    ``declared_size`` is whatever size the adapter already knows, if any. From REST it
    is Starlette's count of the already-buffered body, not a header, so rejecting on it
    saves the DB row and disk write, not the transfer. The limit that counts is the
    one enforced by reading ``stream`` once before anything is stored, so ``stream``
    must be seekable.
    """

    # The name on the client's machine: provenance only, never a storage path.
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
