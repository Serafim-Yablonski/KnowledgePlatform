"""Unit tests for SearchService."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.search import SearchResult, SearchResults
from src.services.search import SearchService


def _make_service(
    repo_results: list[SearchResult] | None = None,
) -> tuple[SearchService, MagicMock, MagicMock]:
    embedding_svc = MagicMock()
    embedding_svc.embed_query = AsyncMock(return_value=[0.1] * 768)

    repo = MagicMock()
    repo.search_similar = AsyncMock(return_value=repo_results or [])

    service = SearchService(
        search_repo=repo,
        embedding_service=embedding_svc,
    )
    return service, embedding_svc, repo


def _result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Test Doc",
        chunk_text="some text",
        score=score,
        chunk_metadata={},
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
