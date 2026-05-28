import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
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


@dataclass
class Cursor:
    created_at: datetime
    id: uuid.UUID


def encode_cursor(cursor: Cursor) -> str:
    payload = json.dumps(
        {"created_at": cursor.created_at.isoformat(), "id": str(cursor.id)}
    )
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(value: str) -> Cursor:
    if len(value) > 512:
        raise ValueError("Invalid pagination cursor")
    try:
        data = json.loads(base64.urlsafe_b64decode(value.encode()))
        return Cursor(
            created_at=datetime.fromisoformat(data["created_at"]),
            id=uuid.UUID(data["id"]),
        )
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid pagination cursor") from exc


@dataclass
class DocumentUpdateInput:
    title: str | None = None


@dataclass
class DocumentPage[T]:
    items: list[T]
    next_cursor: Cursor | None
    has_more: bool
