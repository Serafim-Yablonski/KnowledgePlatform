"""The app as a server runs it once its lifespan has yielded state (ADR-033).

``ASGITransport`` runs no lifespan and sends no ``state``: this does what uvicorn and
Starlette's ``TestClient`` do with lifespan state, a shallow copy into each request's
scope. The real lifespan's state is tested in ``tests/integration/test_lifespan.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send


class ServedWithState:
    """``app`` with ``state`` in every request's scope; usable as a middleware too
    (``app.add_middleware(ServedWithState, state=...)``)."""

    def __init__(self, app: ASGIApp, state: Mapping[str, Any]) -> None:
        self._app = app
        self._state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            scope["state"] = dict(self._state)
        await self._app(scope, receive, send)
