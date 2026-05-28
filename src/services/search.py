from __future__ import annotations

import hashlib
import time
import uuid

import logfire
import structlog

from src.ai.embeddings import EmbeddingService
from src.core.cache import ResponseCache, cached
from src.domain.search import SearchResults
from src.repositories.protocols import SearchRepositoryProtocol

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
    ) -> SearchResults:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        with logfire.span(
            "search",
            workspace_id=str(workspace_id),
            query_hash=query_hash,
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

        return SearchResults(results=results, query=query, total_results=len(results))
