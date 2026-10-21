"""Unit tests for the application factory's wiring (no DB, Redis or network)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.main
import src.mcp_server.auth
from src.core.config import settings
from src.main import create_app


@asynccontextmanager
async def _no_checkpointer() -> AsyncGenerator[None]:
    yield None


@pytest.fixture
def offline_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the lifespan steps that connect out or configure global telemetry."""
    monkeypatch.setattr(src.main, "setup_checkpointer", _no_checkpointer)
    monkeypatch.setattr(src.main, "setup_observability", lambda *a, **kw: None)
    monkeypatch.setattr(src.main, "setup_logging", lambda **kw: None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("offline_lifespan")
async def test_lifespan_runs_its_own_apps_mcp_session_manager() -> None:
    # A session manager runs once per instance, so each app must own its manager
    # and its lifespan must run that one — not the last app's (tests build many).
    first, second = create_app(), create_app()

    async with first.router.lifespan_context(first):
        pass

    with pytest.raises(RuntimeError, match="once per instance"):
        async with first.state.mcp_session_manager.run():
            pass
    async with second.state.mcp_session_manager.run():
        pass


@pytest.fixture
def server_spans() -> Iterator[Callable[[], list[ReadableSpan]]]:
    """Server spans finished while the test runs, from the tracer provider the
    session's Logfire config installed."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]
    yield lambda: [
        s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER
    ]
    processor.shutdown()


@pytest.mark.usefixtures("offline_lifespan")
def test_rest_requests_are_traced_after_startup(
    server_spans: Callable[[], list[ReadableSpan]],
) -> None:
    # Starlette builds its middleware stack on the first ASGI call — the lifespan's —
    # so instrumentation added during startup would never run.
    caller_trace = "0af7651916cd43dd8448eb211c80319c"

    with TestClient(create_app()) as client:  # entered: the lifespan runs first
        client.get(
            "/no-such-route",
            headers={"traceparent": f"00-{caller_trace}-b7ad6b7169203331-01"},
        )

    [span] = server_spans()
    assert format(span.context.trace_id, "032x") != caller_trace


async def _seen_by_app(client_ip: str) -> dict[str, str | None]:
    """Scheme and client IP a route sees for a request forwarded by ``client_ip``."""
    app = create_app()

    async def _echo(request: Request) -> dict[str, str | None]:
        return {
            "scheme": request.url.scheme,
            "client": request.client.host if request.client else None,
        }

    app.add_api_route("/_echo", _echo)
    transport = ASGITransport(app, client=(client_ip, 1234))
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.get(
            "/_echo",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.7"},
        )
    return response.json()  # type: ignore[no-any-return]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trusted_ips",
    ["127.0.0.1", "10.0.0.1, 127.0.0.1", "127.0.0.0/8", "*"],
    ids=["single", "list", "cidr", "any"],
)
async def test_trusted_proxy_headers_applied(
    settings_override: pytest.MonkeyPatch, trusted_ips: str
) -> None:
    # Every form TRUSTED_PROXY_IPS documents: comma-separated IPs, CIDR, "*".
    settings_override.setattr(settings, "TRUSTED_PROXY_HEADERS", True)
    settings_override.setattr(settings, "TRUSTED_PROXY_IPS", trusted_ips)

    seen = await _seen_by_app("127.0.0.1")

    assert seen == {"scheme": "https", "client": "198.51.100.7"}


@pytest.mark.asyncio
async def test_untrusted_proxy_headers_ignored(
    settings_override: pytest.MonkeyPatch,
) -> None:
    settings_override.setattr(settings, "TRUSTED_PROXY_HEADERS", True)
    settings_override.setattr(settings, "TRUSTED_PROXY_IPS", "10.0.0.1, 127.0.0.0/8")

    seen = await _seen_by_app("203.0.113.9")

    assert seen == {"scheme": "http", "client": "203.0.113.9"}


def test_mcp_route_never_hands_a_websocket_to_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MCPAuthMiddleware handles http scopes only; the router must close a websocket
    # to /mcp before the middleware sees it.
    async def _must_not_run(*args: object) -> None:
        raise AssertionError("MCP auth saw a websocket scope")

    monkeypatch.setattr(src.mcp_server.auth, "_authenticate_request", _must_not_run)

    client = TestClient(create_app())  # not entered: no lifespan, so no DB or Redis

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/mcp"):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE", "PUT"])
async def test_mcp_route_passes_every_method_to_mcp(method: str) -> None:
    # The router must not answer 405 itself: method handling is the MCP layer's job.
    # Without a token, MCP auth answers 401 before the transport (no DB needed). The
    # 2026 version header gets past the legacy-protocol guard, whose own 405 would
    # be indistinguishable from the router's.
    transport = ASGITransport(create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.request(
            method, "/mcp", headers={"MCP-Protocol-Version": "2026-07-28"}
        )

    assert response.status_code == 401
