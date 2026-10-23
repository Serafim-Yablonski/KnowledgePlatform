from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.domain.search import SearchQuery


class SearchRequest(BaseModel):
    query: SearchQuery
    top_k: Annotated[int, Field(default=5, ge=1, le=20)] = 5
    min_score: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    section: str | None
    chunk_text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    total_results: int
