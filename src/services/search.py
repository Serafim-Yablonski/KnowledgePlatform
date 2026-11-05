from __future__ import annotations

import hashlib
import math
import time
import uuid

import logfire
import structlog

from src.ai.embeddings import (
    EmbeddingProviderError,
    EmbeddingRateLimited,
    EmbeddingService,
)
from src.core.exceptions import ServiceUnavailableError
from src.domain.search import SearchResults
from src.repositories.protocols import SearchRepositoryProtocol

logger = structlog.get_logger(__name__)


class SearchService:
    def __init__(
        self,
        search_repo: SearchRepositoryProtocol,
        embedding_service: EmbeddingService,
    ) -> None:
        self._repo = search_repo
        self._embedding = embedding_service

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
            embedding = await _embed(self._embedding, query)

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


async def _embed(embedding: EmbeddingService, query: str) -> list[float]:
    """A provider failure is a 503 the caller can act on, not a 500: a REST client
    reads Retry-After, an MCP client's model reads the detail. The provider's own
    message stays in the log."""
    try:
        return await embedding.embed_query(query)
    except EmbeddingRateLimited as exc:
        logger.warning("search embedding rate limited", retry_after=exc.retry_after)
        raise ServiceUnavailableError(
            "Search is temporarily unavailable: the embedding provider is rate limited",
            retry_after=None if exc.retry_after is None else math.ceil(exc.retry_after),
        ) from exc
    except EmbeddingProviderError as exc:
        logger.error("search embedding failed", error_type=type(exc).__name__)
        raise ServiceUnavailableError(
            "Search is temporarily unavailable: the embedding provider failed"
        ) from exc
