import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from src.domain.roles import WorkspaceRole


class WorkspaceStatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    total_tokens_indexed: int
    last_document_updated_at: datetime | None


class WorkspaceCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    description: Annotated[str, Field(max_length=500)] | None = None


class WorkspaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    description: str | None
    is_active: bool
    created_at: datetime
    member_count: int


class WorkspaceMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    email: str
    display_name: str | None
    role: WorkspaceRole
    joined_at: datetime


class AddMemberRequest(BaseModel):
    user_email: EmailStr
    role: WorkspaceRole = WorkspaceRole.MEMBER
