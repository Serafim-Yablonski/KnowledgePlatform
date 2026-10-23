"""Output models of the MCP tools.

A tool's return model is published as its output schema and sent twice: as the
result's structured content and, serialised, as its text block. Clients differ in
which one reaches their model, and both carry the same data, so these are the
model's view of our data whatever the client: MCP's own, not REST's response
schemas, so neither contract changes with the other.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.domain.documents import ContentType, DocumentStatus


class WorkspaceSummary(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = Field(description="What the workspace holds, if set")


class WorkspaceList(BaseModel):
    workspaces: list[WorkspaceSummary]


class SearchHit(BaseModel):
    document_id: uuid.UUID
    document_title: str
    chunk_index: int = Field(
        description="Position of this chunk in the document, from 0"
    )
    section: str | None = Field(description="Heading path in a markdown document")
    chunk_text: str
    score: float = Field(
        description="Cosine similarity to the query, 0-1; higher is closer"
    )


class SearchHits(BaseModel):
    results: list[SearchHit] = Field(description="Best match first")


class DocumentSummary(BaseModel):
    id: uuid.UUID
    title: str
    content_type: ContentType
    status: DocumentStatus = Field(
        description="Only indexed documents are searchable; failed ones never will be"
    )
    created_at: datetime = Field(description="Upload time")


class DocumentList(BaseModel):
    documents: list[DocumentSummary] = Field(description="Newest first")
    next_cursor: str | None = Field(
        description="Pass as cursor for the next page; null on the last page"
    )


class ChunkContent(BaseModel):
    chunk_index: int = Field(description="Position in the document, from 0")
    section: str | None = Field(description="Heading path in a markdown document")
    chunk_text: str


class DocumentContent(BaseModel):
    document_id: uuid.UUID
    title: str
    total_chunks: int
    chunks: list[ChunkContent] = Field(
        description="In document order; neighbours repeat a little text at the seams"
    )
    next_start_chunk: int | None = Field(
        description="Pass as start_chunk to read on; null after the last chunk"
    )
