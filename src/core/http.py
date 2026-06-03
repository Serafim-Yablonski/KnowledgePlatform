from __future__ import annotations

import httpx

_http_client: httpx.AsyncClient | None = None


async def init_http_client() -> httpx.AsyncClient:
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def get_async_http_client() -> httpx.AsyncClient:
    """Return the shared httpx client (initialized by init_http_client).

    Use outside of FastAPI request context, e.g. in MCP tool functions.
    """
    if _http_client is None:
        raise RuntimeError(
            "HTTP client not initialized — call init_http_client() first"
        )
    return _http_client
