from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import ChunkText
from src.models.chunk import DocumentChunk

# As in search: plaintext/PDF chunks carry no section, markdown's preamble carries "".
_SECTION = sa.func.nullif(DocumentChunk.metadata_["section"].as_string(), "")


class SQLAlchemyChunkRepository:
    """Reads one version of a document's chunks; the caller scopes the document."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read_page(
        self, document_id: uuid.UUID, *, version: int, start: int, limit: int
    ) -> list[ChunkText]:
        rows = await self._session.execute(
            sa.select(DocumentChunk.chunk_index, _SECTION, DocumentChunk.text)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.version == version,
                DocumentChunk.chunk_index >= start,
            )
            .order_by(DocumentChunk.chunk_index)
            .limit(limit)
        )
        return [
            ChunkText(chunk_index=index, section=section, chunk_text=text)
            for index, section, text in rows.all()
        ]

    async def count(self, document_id: uuid.UUID, *, version: int) -> int:
        total = await self._session.scalar(
            sa.select(sa.func.count()).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.version == version,
            )
        )
        return total or 0
