from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    score: float
    chunk_metadata: dict[str, Any] = field(default_factory=dict)
