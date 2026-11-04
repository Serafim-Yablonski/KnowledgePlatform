"""Unit tests for EmbeddingService — mocks httpx."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.ai.embeddings import (
    EmbeddingProviderError,
    EmbeddingRateLimited,
    EmbeddingRejected,
    EmbeddingService,
    EmbeddingUnavailable,
)

_DIMS = 768
_MODEL = "gemini-embedding-001"
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

    async def fake_post(url: str, json: Any, **_: Any) -> _MockResponse:  # type: ignore[override]
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

    async def fake_post(url: str, json: Any, **_: Any) -> _MockResponse:  # type: ignore[override]
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


def _rate_limited(retry_delay: str | None = None) -> httpx.Response:
    """Gemini's 429: a google.rpc.Status whose details may carry a RetryInfo."""
    details: list[dict[str, Any]] = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}
    ]
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        },
    )


async def _embed_with(
    replies: list[httpx.Response | Exception],
) -> tuple[list[float], int, list[list[float]] | Exception]:
    """Embed one text against scripted replies; return the sleeps, the number of
    calls and the result or the raised error."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        reply = replies[min(calls, len(replies) - 1)]
        calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply

    sleep = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = EmbeddingService(
            api_key=_API_KEY, model=_MODEL, dimensions=_DIMS, http_client=client
        )
        with patch("src.ai.embeddings.asyncio.sleep", sleep):
            try:
                outcome: list[list[float]] | Exception = await service.embed_texts(
                    ["t"]
                )
            except EmbeddingProviderError as exc:
                outcome = exc
    return [c.args[0] for c in sleep.await_args_list], calls, outcome


_OK = httpx.Response(200, json=_gemini_response(["t"]))


@pytest.mark.parametrize(
    ("retry_delay", "slept"), [("7s", 7.0), ("2.5s", 2.5), ("0.2s", 1.0)]
)
async def test_a_rate_limit_waits_the_providers_delay_or_our_backoff_if_longer(
    retry_delay: str, slept: float
) -> None:
    # The provider's delay is a hint that can be too short ("1s" while still
    # limited): the first own backoff (1 s) is the floor.
    sleeps, calls, outcome = await _embed_with([_rate_limited(retry_delay), _OK])

    assert sleeps == [slept]
    assert calls == 2
    assert isinstance(outcome, list)


async def test_a_rate_limit_too_long_to_wait_gives_up_at_once_with_the_delay() -> None:
    # A daily quota asks for hours: neither a search request nor a worker attempt
    # can wait that out, so the caller gets the delay to act on.
    sleeps, calls, outcome = await _embed_with([_rate_limited("3600s")])

    assert isinstance(outcome, EmbeddingRateLimited)
    assert outcome.retry_after == 3600.0
    assert (sleeps, calls) == ([], 1)


async def test_a_lasting_rate_limit_raises_without_sleeping_after_the_last_try() -> (
    None
):
    sleeps, calls, outcome = await _embed_with([_rate_limited()])

    assert isinstance(outcome, EmbeddingRateLimited)
    assert outcome.retry_after is None
    assert (sleeps, calls) == ([1.0, 2.0], 3)


@pytest.mark.parametrize(
    "body", [{"error": "rate limited"}, {"error": {"details": [{"retryDelay": 5}]}}, []]
)
async def test_a_rate_limit_with_an_unreadable_delay_uses_our_backoff(
    body: Any,
) -> None:
    sleeps, _, outcome = await _embed_with([httpx.Response(429, json=body), _OK])

    assert sleeps == [1.0]
    assert isinstance(outcome, list)


async def test_a_server_error_that_persists_raises_unavailable() -> None:
    sleeps, calls, outcome = await _embed_with([httpx.Response(503, text="down")])

    assert isinstance(outcome, EmbeddingUnavailable)
    assert (sleeps, calls) == ([1.0, 1.0], 3)


async def test_a_network_error_raises_unavailable() -> None:
    _, calls, outcome = await _embed_with([httpx.ConnectError("refused")])

    assert isinstance(outcome, EmbeddingUnavailable)
    assert calls == 1


async def test_a_failed_batch_cancels_the_others_and_raises_its_own_error() -> None:
    # A batch left running would call the provider for an attempt that already
    # failed, overlapping the next attempt's calls.
    cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if len(json.loads(request.content)["requests"]) == 100:
            return httpx.Response(400, text="no")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = EmbeddingService(
            api_key=_API_KEY, model=_MODEL, dimensions=_DIMS, http_client=client
        )
        with pytest.raises(EmbeddingRejected):
            await service.embed_texts(["t"] * 150)

        assert cancelled.is_set()


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_a_client_error_raises_rejected_without_retry(status: int) -> None:
    sleeps, calls, outcome = await _embed_with([httpx.Response(status, text="no")])

    assert isinstance(outcome, EmbeddingRejected)
    assert outcome.status_code == status
    assert (sleeps, calls) == ([], 1)


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector() -> None:
    service = _make_service()

    async def fake_post(url: str, json: Any, **_: Any) -> _MockResponse:  # type: ignore[override]
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


async def test_api_key_is_sent_as_a_header_not_in_the_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A URL is recorded wherever the request is traced or logged (httpx logs each
    # request's URL at INFO); the key must still authenticate every call.
    caplog.set_level(logging.INFO, logger="httpx")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_gemini_response(["query"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = EmbeddingService(
            api_key=_API_KEY, model=_MODEL, dimensions=_DIMS, http_client=client
        )
        await service.embed_query("query")

    [request] = requests
    assert _API_KEY not in str(request.url)
    assert request.headers["x-goog-api-key"] == _API_KEY
    assert "HTTP Request: POST" in caplog.text
    assert _API_KEY not in caplog.text


@pytest.mark.parametrize(
    ("embedding_key", "google_key", "sent"),
    [("embed-key", "google-key", "embed-key"), ("", "google-key", "google-key")],
    ids=["own-key", "falls-back-to-google"],
)
async def test_the_configured_client_sends_the_embedding_key_first(
    embedding_key: str, google_key: str, sent: str
) -> None:
    """The API and the workers build their client from the same settings."""
    from src.core.config import AISettings

    cfg = AISettings(EMBEDDING_API_KEY=embedding_key, GOOGLE_API_KEY=google_key)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-goog-api-key"])
        return httpx.Response(200, json={"embeddings": [{"values": _fake_embedding()}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = EmbeddingService.from_settings(cfg, client)
        await service.embed_texts(["t"])

    assert seen == [sent]


async def _posted_url(service_for: Any) -> str:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"embeddings": [{"values": _fake_embedding()}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await service_for(client).embed_texts(["t"])
    [url] = urls
    return url


@pytest.mark.parametrize(
    "base_url", ["", "https://generativelanguage.googleapis.com/v1beta"]
)
async def test_requests_go_to_gemini_unless_configured(base_url: str) -> None:
    from src.core.config import AISettings

    cfg = AISettings(EMBEDDING_BASE_URL=base_url, EMBEDDING_MODEL=_MODEL)
    url = await _posted_url(lambda client: EmbeddingService.from_settings(cfg, client))

    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-embedding-001:batchEmbedContents"
    )


@pytest.mark.parametrize(
    "base_url", ["http://127.0.0.1:8089/v1beta", "http://127.0.0.1:8089/v1beta/"]
)
async def test_requests_go_to_the_configured_base_url(base_url: str) -> None:
    """A fake provider in the acceptance suite, a proxy or a regional endpoint."""
    from src.core.config import AISettings

    cfg = AISettings(EMBEDDING_BASE_URL=base_url, EMBEDDING_MODEL=_MODEL)
    url = await _posted_url(lambda client: EmbeddingService.from_settings(cfg, client))

    assert url == (
        "http://127.0.0.1:8089/v1beta/models/gemini-embedding-001:batchEmbedContents"
    )
