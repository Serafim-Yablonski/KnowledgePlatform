"""Unit tests for EmbeddingService — mocks httpx."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.embeddings import EmbeddingService

_DIMS = 768
_MODEL = "text-embedding-005"
_API_KEY = "test-key"


def _fake_embedding(seed: int = 0) -> list[float]:
    """Return a deterministic 768-dim unit vector for testing."""
    v = [float(seed % 10 + i % 5) for i in range(_DIMS)]
    return v


def _make_service() -> EmbeddingService:
    return EmbeddingService(
        api_key=_API_KEY,
        model=_MODEL,
        dimensions=_DIMS,
    )


def _gemini_response(texts: list[str]) -> dict[str, Any]:
    return {"embeddings": [{"values": _fake_embedding(i)} for i in range(len(texts))]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_splitting_250_texts_makes_three_api_calls() -> None:
    """250 texts → 3 batches (100 + 100 + 50) because Gemini limit is 100."""
    texts = [f"text {i}" for i in range(250)]
    service = _make_service()

    call_sizes: list[int] = []

    async def fake_post(url: str, json: Any) -> _MockResponse:  # type: ignore[override]
        n = len(json["requests"])
        call_sizes.append(n)
        return _MockResponse(200, _gemini_response(["x"] * n))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        results = await service.embed_texts(texts)

    assert call_sizes == [100, 100, 50]
    assert len(results) == 250


@pytest.mark.asyncio
async def test_retry_on_429_then_success() -> None:
    """First call returns 429; second call succeeds — verify retry logic."""
    service = _make_service()
    call_count = 0

    async def fake_post(url: str, json: Any) -> _MockResponse:  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _MockResponse(429, {"error": "rate limited"})
        return _MockResponse(200, _gemini_response(["hi"]))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            results = await service.embed_texts(["hi"])

    assert call_count == 2
    assert len(results) == 1
    assert len(results[0]) == _DIMS


@pytest.mark.asyncio
async def test_4xx_raises_immediately_without_retry() -> None:
    """A 400 error raises RuntimeError immediately — no retries."""
    service = _make_service()
    call_count = 0

    async def fake_post(url: str, json: Any) -> _MockResponse:  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        return _MockResponse(400, {"error": "bad request"})

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="400"):
            await service.embed_texts(["bad"])

    assert call_count == 1


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector() -> None:
    service = _make_service()

    async def fake_post(url: str, json: Any) -> _MockResponse:  # type: ignore[override]
        return _MockResponse(200, _gemini_response(["query"]))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        result = await service.embed_query("query")

    assert isinstance(result, list)
    assert len(result) == _DIMS
