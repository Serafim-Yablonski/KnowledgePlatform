from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

# gemini-embedding-001 reads 2,048 tokens and silently drops the rest; 2,000
# characters stay within that up to 1 token per character, the densest input our
# clients send (IDs, hashes, emoji). Re-derive on a provider change. ADR-022.
SearchQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    chunk_index: int  # position in the document's current version, from 0
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    score: float
    section: str | None  # markdown heading path; None for other formats


class SearchResults(BaseModel):
    results: list[SearchResult]
    query: str
    total_results: int

    model_config = ConfigDict(arbitrary_types_allowed=True)
