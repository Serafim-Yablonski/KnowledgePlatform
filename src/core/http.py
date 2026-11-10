from __future__ import annotations

import httpx


def build_http_client() -> httpx.AsyncClient:
    """The API process's shared outbound HTTP client; its lifespan builds one
    (ADR-033)."""
    return httpx.AsyncClient(timeout=30.0)
