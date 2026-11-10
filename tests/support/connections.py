"""What the tests observe (ADR-033): held connections, the embedding provider's
calls, and any other outbound HTTP call."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from src.models.chunk import EMBEDDING_DIMS

VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)


class HeldConnections:
    """Connections checked out of an engine's pool right now, and the peak, counted
    from the pool's checkout and checkin events."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0

    def checkout(self, *_: Any) -> None:
        self.now += 1
        self.peak = max(self.peak, self.now)

    def checkin(self, *_: Any) -> None:
        self.now -= 1


class EmbeddingProvider:
    """Gemini's batchEmbedContents, in-process behind ``httpx.MockTransport``.

    Answers every text with ``VECTOR``; ``on_request`` runs inside each call, while
    the caller is waiting on the provider.
    """

    def __init__(self) -> None:
        self.on_request: Callable[[], None] = lambda: None

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.on_request()
        texts = json.loads(request.content)["requests"]
        return httpx.Response(
            200, json={"embeddings": [{"values": VECTOR}] * len(texts)}
        )


class Outbound:
    """The app's outbound HTTP peer, behind ``httpx.MockTransport``: a call fails the
    test unless the test sets ``handler``."""

    def __init__(self) -> None:
        self.handler: Callable[[httpx.Request], httpx.Response] | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.handler is None:
            raise AssertionError(f"unexpected outbound HTTP call: {request.url}")
        return self.handler(request)
