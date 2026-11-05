from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import DocumentStatus
from src.domain.search import SearchResult
from src.models.chunk import EMBEDDING_DIMS

# Exact search over the workspace's chunks (ADR-029). The score filter sits outside
# the LIMIT: beside the workspace filter, PostgreSQL pushes it into the chunk scan,
# which then computes the distance for every chunk of every workspace. Applied to
# the top k it gives the same rows, as they are ranked by that same score.
SIMILAR_CHUNKS_SQL = f"""
    SELECT * FROM (
        SELECT
            c.id,
            c.chunk_index,
            c.document_id,
            d.title  AS document_title,
            c.text,
            1 - (c.embedding <=> (:query_embedding)::vector({EMBEDDING_DIMS}))
                     AS similarity_score,
            NULLIF(c.metadata_ ->> 'section', '') AS section
        FROM  document_chunks c
        JOIN  documents d ON c.document_id = d.id
        WHERE d.workspace_id = :workspace_id
          AND d.status       = :status
        ORDER BY similarity_score DESC
        LIMIT :top_k
    ) AS nearest
    WHERE  similarity_score >= :min_score
    ORDER  BY similarity_score DESC
"""


class SQLAlchemySearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_similar(
        self,
        workspace_id: uuid.UUID,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        # asyncpg requires pgvector codec registration to bind a list as vector,
        # which we avoid by passing the embedding as a plain text string.
        # PostgreSQL casts it server-side via (:param)::vector(N).
        # Wrapping in parentheses is necessary: SQLAlchemy's asyncpg dialect
        # skips `:name::type` patterns (treats `::` as a cast and leaves the
        # leading `:name` unparsed as a positional parameter).
        # str(float(x)) only produces digits/dots/signs — no injection risk.
        vec_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        rows = (
            await self._session.execute(
                sa.text(SIMILAR_CHUNKS_SQL),
                {
                    "query_embedding": vec_str,
                    "workspace_id": workspace_id,
                    # SQLAlchemy's non-native Enum stores the member NAME (uppercase),
                    # not the StrEnum value. Confirmed against live PostgreSQL 18.
                    "status": DocumentStatus.INDEXED.name,
                    "top_k": top_k,
                    "min_score": min_score,
                },
            )
        ).all()

        return [
            SearchResult(
                chunk_id=row.id,
                chunk_index=row.chunk_index,
                document_id=row.document_id,
                document_title=row.document_title,
                chunk_text=row.text,
                score=float(row.similarity_score),
                section=row.section,
            )
            for row in rows
        ]
