from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class WorkspaceStats:
    document_count: int
    chunk_count: int
    total_tokens_indexed: int
    last_document_updated_at: datetime | None
