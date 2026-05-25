import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]


class ApiKeyCreateResponse(BaseModel):
    id: uuid.UUID
    key: str
    prefix: str
    name: str


class ApiKeyListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prefix: str
    name: str
    last_used_at: datetime | None
    is_active: bool
    created_at: datetime
