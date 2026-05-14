from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chunk import DocumentChunk


class SQLAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_batch(self, chunks: list[DocumentChunk]) -> None:
        self._session.add_all(chunks)
        await self._session.flush()

    async def delete_by_document_version(
        self, document_id: uuid.UUID, below_version: int
    ) -> int:
        result = await self._session.execute(
            sa.delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.version < below_version,
            )
        )
        return result.rowcount  # type: ignore[attr-defined, no-any-return]

    async def get_by_document(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        result = await self._session.execute(
            sa.select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())
