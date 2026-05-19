from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(default=5, ge=1, le=20)] = 5
    min_score: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_text: str
    document_id: uuid.UUID
    document_title: str
    score: float
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    total_results: int
