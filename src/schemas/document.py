import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.domain.documents import ContentType, DocumentStatus


class DocumentCreate(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=255)]


class DocumentUpdate(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=255)] | None = None


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    content_type: ContentType
    status: DocumentStatus
    version: int
    file_size_bytes: int
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class PaginatedResponse[T](BaseModel):
    items: list[T]
    next_cursor: str | None
    has_more: bool
