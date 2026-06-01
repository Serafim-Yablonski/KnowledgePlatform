from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    score: float
    chunk_metadata: dict[str, Any] = field(default_factory=dict)


class SearchResults(BaseModel):
    results: list[SearchResult]
    query: str
    total_results: int

    model_config = ConfigDict(arbitrary_types_allowed=True)
