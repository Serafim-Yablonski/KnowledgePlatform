from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from src.domain.roles import WorkspaceRole


@dataclass
class WorkspaceStats:
    document_count: int
    chunk_count: int
    total_tokens_indexed: int
    last_document_updated_at: datetime | None


@dataclass
class WorkspaceInfo:
    id: uuid.UUID
    name: str
    slug: str
    description: str | None
    is_active: bool
    created_at: datetime
    member_count: int


@dataclass
class WorkspaceMember:
    user_id: uuid.UUID
    email: str
    display_name: str | None
    role: WorkspaceRole
    joined_at: datetime
