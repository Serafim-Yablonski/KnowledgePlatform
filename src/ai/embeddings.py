"""Google Gemini embedding service.

Switching to OpenAI or Voyage requires changing the URL, request body format, and
batch limit. All documents must be re-embedded after switching — trigger via the
re-indexing pipeline (bump Document.version and re-dispatch embed_chunks tasks).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import logfire
import structlog

logger = structlog.get_logger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_BATCH_LIMIT = 100  # Gemini batchEmbedContents limit (NOT 2048 like OpenAI)


class EmbeddingService:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._url = f"{_GEMINI_BASE}/{model}:batchEmbedContents?key={api_key}"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with logfire.span(
            "embed_texts",
            model=self._model,
            dimensions=self._dimensions,
            embedding_batch_size=len(texts),
        ):
            return await self._fetch_embeddings(texts)

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call Gemini batchEmbedContents in batches of _BATCH_LIMIT."""
        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for batch_start in range(0, len(texts), _BATCH_LIMIT):
                batch = texts[batch_start : batch_start + _BATCH_LIMIT]
                embeddings = await self._call_api(client, batch)
                all_embeddings.extend(embeddings)

        return all_embeddings

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

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.post(self._url, json=payload)
            except httpx.RequestError as exc:
                raise RuntimeError(f"Embedding API request failed: {exc}") from exc

            if response.status_code == 200:
                data = response.json()
                return [e["values"] for e in data["embeddings"]]

            if response.status_code == 429:
                wait = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "embedding rate limited, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
                last_exc = RuntimeError(f"Embedding API rate limited: {response.text}")
                continue

            if response.status_code >= 500:
                last_exc = RuntimeError(
                    f"Embedding API server error {response.status_code}: "
                    f"{response.text}"
                )
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                raise last_exc

            # 4xx errors are not retried.
            raise RuntimeError(
                f"Embedding API client error {response.status_code}: {response.text}"
            )

        raise RuntimeError("Embedding API failed after retries") from last_exc
