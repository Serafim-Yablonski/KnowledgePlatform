import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.config import settings
from src.core.exceptions import InputValidationError


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ContentType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    PLAINTEXT = "plaintext"


MAX_UPLOAD_SIZE_BYTES: int = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

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
        raise InputValidationError("Invalid pagination cursor")
    try:
        data = json.loads(base64.urlsafe_b64decode(value.encode()))
        return Cursor(
            created_at=datetime.fromisoformat(data["created_at"]),
            id=uuid.UUID(data["id"]),
        )
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        raise InputValidationError("Invalid pagination cursor") from exc
