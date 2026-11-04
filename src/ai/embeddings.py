"""Google Gemini embedding service.

Switching to OpenAI or Voyage requires changing the URL, request body format, and
batch limit. All documents must be re-embedded after switching: ``make reindex``
(ADR-030) owes each one embedding again, which replaces its chunk set.
"""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

import httpx
import logfire
import structlog

from src.core.config import AISettings

logger = structlog.get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_BATCH_LIMIT = 100  # Gemini batchEmbedContents limit (NOT 2048 like OpenAI)
_TRIES = 3
# A 429 asking for longer than this (a daily quota asks for hours) is raised at
# once with its delay: neither a search request nor a worker attempt waits it out.
_MAX_RETRY_WAIT_S = 10.0


class EmbeddingProviderError(Exception):
    """The embedding provider didn't return embeddings."""


class EmbeddingRateLimited(EmbeddingProviderError):
    """429. ``retry_after`` is the provider's delay in seconds, when it gave one."""

    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingUnavailable(EmbeddingProviderError):
    """A 5xx, or no response at all."""


class EmbeddingRejected(EmbeddingProviderError):
    """Any other 4xx: the provider refused the request itself."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmbeddingService:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = GEMINI_BASE_URL,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._url = f"{base_url.rstrip('/')}/models/{model}:batchEmbedContents"
        # In a header, not the URL's ?key=: traces and httpx's request log record
        # the full URL, never request headers.
        self._headers = {"x-goog-api-key": api_key}
        # Shared client injected by its owner (the app's lifespan, a worker's
        # process); None: a client per call.
        self._http_client = http_client

    @classmethod
    def from_settings(
        cls, cfg: AISettings, http_client: httpx.AsyncClient | None = None
    ) -> EmbeddingService:
        """The one configured embedding client, for the API and the workers."""
        return cls(
            api_key=(
                cfg.EMBEDDING_API_KEY.get_secret_value()
                or cfg.GOOGLE_API_KEY.get_secret_value()
            ),
            model=cfg.EMBEDDING_MODEL,
            dimensions=cfg.EMBEDDING_DIMENSIONS,
            http_client=http_client,
            base_url=cfg.EMBEDDING_BASE_URL or GEMINI_BASE_URL,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with logfire.span(
            "embed_texts",
            model=self._model,
            dimensions=self._dimensions,
            embedding_batch_size=len(texts),
        ) as span:
            t0 = monotonic()
            result = await self._fetch_embeddings(texts)
            span.set_attribute(
                "embedding_api_duration_ms", round((monotonic() - t0) * 1000)
            )
            return result

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call Gemini batchEmbedContents with all batches in parallel."""
        batches = [
            texts[i : i + _BATCH_LIMIT] for i in range(0, len(texts), _BATCH_LIMIT)
        ]
        if self._http_client is not None:
            results = await self._call_all(self._http_client, batches)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                results = await self._call_all(client, batches)
        return [embedding for batch_result in results for embedding in batch_result]

    async def _call_all(
        self, client: httpx.AsyncClient, batches: list[list[str]]
    ) -> list[list[list[float]]]:
        """Every batch's embeddings. One failing cancels the rest, so no request
        outlives the call; the caller gets the first failure as itself."""
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._call_api(client, batch))
                    for batch in batches
                ]
        except ExceptionGroup as failed:
            # The cancelled batches aren't in it: the first entry failed first.
            raise failed.exceptions[0] from None
        return [task.result() for task in tasks]

    async def _call_api(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> list[list[float]]:
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self._dimensions,
                }
                for t in texts
            ]
        }

        for attempt in range(_TRIES):
            last_try = attempt == _TRIES - 1
            try:
                response = await client.post(
                    self._url, json=payload, headers=self._headers
                )
            except httpx.RequestError as exc:
                raise EmbeddingUnavailable(
                    f"Embedding API request failed: {exc}"
                ) from exc

            if response.status_code == 200:
                data = response.json()
                return [e["values"] for e in data["embeddings"]]

            if response.status_code == 429:
                delay = _retry_delay(response)
                wait = max(delay or 0.0, float(2**attempt))  # own backoff: 1s, 2s
                if last_try or wait > _MAX_RETRY_WAIT_S:
                    raise EmbeddingRateLimited(
                        f"Embedding API rate limited: {response.text}", delay
                    )
                logger.warning(
                    "embedding rate limited, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code >= 500:
                if last_try:
                    raise EmbeddingUnavailable(
                        f"Embedding API server error {response.status_code}: "
                        f"{response.text}"
                    )
                await asyncio.sleep(1)
                continue

            # Other 4xx: the same request gets the same answer.
            raise EmbeddingRejected(
                f"Embedding API client error {response.status_code}: {response.text}",
                response.status_code,
            )

        raise AssertionError("unreachable: the last try returns or raises")


def _retry_delay(response: httpx.Response) -> float | None:
    """The delay a 429 asks for: a google.rpc.RetryInfo in the error's details,
    whose ``retryDelay`` is a protobuf Duration ("34s", "40.15s"). None when the
    body carries none we can read."""
    try:
        details = response.json()["error"]["details"]
        for detail in details:
            if detail.get("@type", "").endswith("google.rpc.RetryInfo"):
                return float(detail["retryDelay"].removesuffix("s"))
    except ValueError, KeyError, TypeError, AttributeError:
        pass
    return None
