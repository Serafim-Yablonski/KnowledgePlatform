from __future__ import annotations

import time
import uuid

import logfire
import structlog

from src.ai.embeddings import EmbeddingService
from src.core.cache import ResponseCache, cached
from src.repositories.protocols import SearchRepositoryProtocol
from src.schemas.search import SearchResponse, SearchResultItem

logger = structlog.get_logger(__name__)


class SearchService:
    def __init__(
        self,
        search_repo: SearchRepositoryProtocol,
        embedding_service: EmbeddingService,
        cache: ResponseCache,
    ) -> None:
        self._repo = search_repo
        self._embedding = embedding_service
        self._cache = cache

    @cached(
        ttl=300,
        key_template="search:{workspace_id}:{query_hash}:{top_k}:{min_score}",
    )
    async def search(
        self,
        workspace_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> SearchResponse:
        with logfire.span(
            "search",
            workspace_id=str(workspace_id),
            query_length=len(query),
            top_k=top_k,
            min_score=min_score,
        ) as span:
            t0 = time.monotonic()
            embedding = await self._embedding.embed_query(query)

            results = await self._repo.search_similar(
                workspace_id=workspace_id,
                embedding=embedding,
                top_k=top_k,
                min_score=min_score,
            )

            search_latency_ms = round((time.monotonic() - t0) * 1000)
            top_score = results[0].score if results else 0.0

            span.set_attribute("result_count", len(results))
            span.set_attribute("top_score", top_score)
            span.set_attribute("search_latency_ms", search_latency_ms)

        logger.info(
            "search completed",
            workspace_id=str(workspace_id),
            result_count=len(results),
            top_score=top_score,
            search_latency_ms=search_latency_ms,
        )

        return SearchResponse(
            results=[
                SearchResultItem(
                    chunk_text=r.chunk_text,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    score=r.score,
                    chunk_metadata=r.chunk_metadata,
                )
                for r in results
            ],
            query=query,
            total_results=len(results),
        )
