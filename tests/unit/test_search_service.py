"""Unit tests for SearchService."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ai.embeddings import (
    EmbeddingRateLimited,
    EmbeddingRejected,
    EmbeddingUnavailable,
)
from src.core.exceptions import ServiceUnavailableError
from src.domain.search import SearchResult, SearchResults
from src.services.search import SearchService
from tests.support.unit_of_work import FakeUnitOfWork


def _make_service(
    repo_results: list[SearchResult] | None = None,
) -> tuple[SearchService, MagicMock, MagicMock]:
    embedding_svc = MagicMock()
    embedding_svc.embed_query = AsyncMock(return_value=[0.1] * 768)

    repo = MagicMock()
    repo.search_similar = AsyncMock(return_value=repo_results or [])

    service = SearchService(FakeUnitOfWork(search=repo), embedding_svc)
    return service, embedding_svc, repo


def _result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        chunk_index=0,
        document_id=uuid.uuid4(),
        document_title="Test Doc",
        chunk_text="some text",
        score=score,
        section=None,
    )


class TestSearchService:
    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        ws_id = uuid.uuid4()
        service, emb, repo = _make_service(repo_results=[_result(0.9)])

        response = await service.search(workspace_id=ws_id, query="test query")

        assert isinstance(response, SearchResults)
        assert len(response.results) == 1
        assert response.query == "test query"
        assert response.total_results == 1

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        ws_id = uuid.uuid4()
        service, emb, repo = _make_service(repo_results=[])

        response = await service.search(workspace_id=ws_id, query="no match")

        assert response.results == []
        assert response.total_results == 0

    @pytest.mark.asyncio
    async def test_results_ordered_by_score_descending(self) -> None:
        ws_id = uuid.uuid4()
        results = [_result(0.9), _result(0.7), _result(0.5)]
        service, emb, repo = _make_service(repo_results=results)

        response = await service.search(workspace_id=ws_id, query="q")

        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_embeds_query(self) -> None:
        ws_id = uuid.uuid4()
        service, emb, repo = _make_service()

        await service.search(workspace_id=ws_id, query="semantic question")

        emb.embed_query.assert_awaited_once_with("semantic question")

    @pytest.mark.asyncio
    async def test_passes_params_to_repo(self) -> None:
        ws_id = uuid.uuid4()
        service, emb, repo = _make_service()

        await service.search(workspace_id=ws_id, query="q", top_k=3, min_score=0.5)

        repo.search_similar.assert_awaited_once_with(
            workspace_id=ws_id,
            embedding=[0.1] * 768,
            top_k=3,
            min_score=0.5,
        )

    @pytest.mark.parametrize(
        ("error", "retry_after"),
        [
            (EmbeddingRateLimited("429", retry_after=29.2), 30),
            (EmbeddingRateLimited("429", retry_after=None), None),
            (EmbeddingUnavailable("503"), None),
            (EmbeddingRejected("403", 403), None),
        ],
        ids=["429-with-delay", "429", "5xx", "4xx"],
    )
    async def test_a_provider_failure_is_a_503_not_a_bug(
        self, error: Exception, retry_after: int | None
    ) -> None:
        # Unmapped, it was a 500 on REST and an opaque error to an MCP client's
        # model, which then can't tell "retry later" from a bug.
        service, emb, repo = _make_service()
        emb.embed_query.side_effect = error

        with pytest.raises(ServiceUnavailableError) as exc_info:
            await service.search(workspace_id=uuid.uuid4(), query="q")

        assert exc_info.value.retry_after == retry_after
        assert "embedding provider" in exc_info.value.detail
        assert str(error) not in exc_info.value.detail
        repo.search_similar.assert_not_awaited()
