from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    document_id: uuid.UUID
    document_title: Annotated[str, Field(max_length=500)]
    chunk_text: Annotated[str, Field(max_length=4000)]
    relevance_score: float


class Answer(BaseModel):
    answer: Annotated[str, Field(max_length=10000)]
    sources: list[SourceReference]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the agent is in this answer",
    )
    reasoning: Annotated[str, Field(max_length=2000)] = Field(
        description="Brief explanation of how the answer was derived",
    )
