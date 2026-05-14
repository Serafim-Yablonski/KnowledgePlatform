"""Unit tests for EmbeddingService — mocks httpx and Redis."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.embeddings import EmbeddingService

_DIMS = 768
_MODEL = "text-embedding-005"
_API_KEY = "test-key"


def _fake_embedding(seed: int = 0) -> list[float]:
    """Return a deterministic 768-dim unit vector for testing."""
    v = [float(seed % 10 + i % 5) for i in range(_DIMS)]
    return v


def _make_redis(cached: dict[str, str] | None = None) -> AsyncMock:
    """Build a minimal async Redis mock with mget + pipeline support."""
    redis = AsyncMock()
    stored: dict[str, str] = dict(cached or {})

    async def mget(*keys: str) -> list[str | None]:
        return [stored.get(k) for k in keys]

    redis.mget = mget

    pipe = AsyncMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    pipe.set = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=None)
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


def _make_service(redis: AsyncMock | None = None) -> EmbeddingService:
    return EmbeddingService(
        api_key=_API_KEY,
        model=_MODEL,
        dimensions=_DIMS,
        redis_client=redis or _make_redis(),
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
async def test_cache_hit_skips_api_call() -> None:
    """embed_texts called twice with the same input hits the API only once."""
    service = _make_service()

    response = _MockResponse(200, _gemini_response(["hello"]))
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=response)
        mock_client_cls.return_value = mock_client

        first = await service.embed_texts(["hello"])

    # Populate cache manually with what we got.
    key = service._cache_key("hello")
    cached_redis = _make_redis(cached={key: json.dumps(first[0])})
    service2 = _make_service(redis=cached_redis)

    with patch("httpx.AsyncClient") as mock_client_cls2:
        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
        mock_client2.__aexit__ = AsyncMock(return_value=False)
        mock_client2.post = AsyncMock()
        mock_client_cls2.return_value = mock_client2

        second = await service2.embed_texts(["hello"])
        mock_client2.post.assert_not_called()

    assert second[0] == first[0]


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
async def test_cache_key_includes_dimensions_so_dim_change_is_cache_miss() -> None:
    """Cache key encodes dimensions — switching EMBEDDING_DIMENSIONS busts the cache."""
    text = "same text"
    service_768 = EmbeddingService(
        api_key=_API_KEY, model=_MODEL, dimensions=768, redis_client=_make_redis()
    )
    service_256 = EmbeddingService(
        api_key=_API_KEY, model=_MODEL, dimensions=256, redis_client=_make_redis()
    )
    key_768 = service_768._cache_key(text)
    key_256 = service_256._cache_key(text)
    assert key_768 != key_256
    assert ":768:" in key_768
    assert ":256:" in key_256


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
