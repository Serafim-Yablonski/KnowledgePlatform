"""Who controls our traces: incoming trace context per process, and import safety.

Tracing config is process-global (Logfire's propagator, structlog, the root
logger), so each test runs its scenario in a fresh interpreter: inside pytest the
modules may already be imported and configured.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.workers.__main__ import PHASES

ROOT = Path(__file__).resolve().parents[2]

# An empty token beats .env's (pydantic-settings reads the environment first), so
# the worker scenario, which configures Logfire from settings, never exports.
_OFFLINE_ENV = {**os.environ, "LOGFIRE_TOKEN": ""}

# probe() uses the MCP SDK's private `_otel` module, the exact path `_meta` takes;
# an SDK upgrade that renames it fails every test here at once, never silently.
# probe() / probe_rest() print how a span started under an incoming traceparent
# turns out: "recording=<bool> adopted=<bool>". probe() takes the path the MCP SDK
# takes for `_meta`; probe_rest() sends the HTTP header through an app's FastAPI
# instrumentation. flags "00" = the caller says "don't sample", "01" = "sample".
# probe_libraries() prints which libraries' calls are traced.
_PROBE = """
from fastapi.testclient import TestClient
from mcp.shared._otel import extract_trace_context, otel_span
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

TRACE = "0af7651916cd43dd8448eb211c80319c"


def _traceparent(flags):
    return f"00-{TRACE}-b7ad6b7169203331-{flags}"


def probe(label, flags="00"):
    ctx = extract_trace_context({"traceparent": _traceparent(flags)})
    with otel_span("probe", kind=SpanKind.SERVER, context=ctx) as span:
        adopted = format(span.get_span_context().trace_id, "032x") == TRACE
        print(f"{label}: recording={span.is_recording()} adopted={adopted}")


def probe_rest(app, label, flags="00"):
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    TestClient(app).get("/no-such-route", headers={"traceparent": _traceparent(flags)})
    spans = [s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER]
    adopted = any(format(s.context.trace_id, "032x") == TRACE for s in spans)
    print(f"{label}: recording={len(spans) == 1} adopted={adopted}")


def probe_libraries(label):
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    instrumentors = {
        "asyncpg": AsyncPGInstrumentor(),
        "httpx": HTTPXClientInstrumentor(),
    }
    traced = [n for n, i in instrumentors.items() if i.is_instrumented_by_opentelemetry]
    print(f"{label}: {' '.join(traced) or '-'}")
"""


def _run(scenario: str, env: dict[str, str] = _OFFLINE_ENV) -> list[str]:
    code = _PROBE + textwrap.dedent(scenario)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith("probe ")]


def test_no_token_exports_nothing_even_with_one_in_the_environment() -> None:
    # Logfire falls back to a LOGFIRE_TOKEN environment variable when it is given
    # none, so "no token" has to switch exporting off explicitly.
    lines = _run(
        """
        import logfire
        from src.core.observability import configure_logfire

        configure_logfire(
            token=None,
            environment="test",
            service_name="kp",
            trust_incoming_trace_context=False,
        )
        print(f"probe send={logfire.DEFAULT_LOGFIRE_INSTANCE.config.send_to_logfire}")
        """,
        # Without pytest's variables: under PYTEST_VERSION Logfire never exports by
        # default, which would hide the fallback.
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")},
            "LOGFIRE_TOKEN": "pylf_v1_us_not_a_real_token",
        },
    )

    assert lines == ["probe send=False"]


def test_api_ignores_client_trace_context() -> None:
    # A client must not choose our trace ids, nor switch recording off with an
    # unsampled traceparent (Logfire's default sampler follows the parent) — over
    # MCP's `_meta` or REST's header. In production order: the app is built (and
    # instrumented) at import, Logfire is configured at startup.
    lines = _run(
        """
        from src.main import app
        from src.core.observability import setup_observability

        setup_observability(token=None, environment="test", service_name="kp")
        probe("probe api mcp")
        probe_rest(app, "probe api rest")
        """
    )

    assert lines == [
        "probe api mcp: recording=True adopted=False",
        "probe api rest: recording=True adopted=False",
    ]


def test_api_and_worker_trace_the_same_libraries() -> None:
    # Database queries and outbound HTTP, in both processes; nothing before setup
    # (importing configures nothing).
    lines = _run(
        """
        import src.main
        import src.workers.__main__
        from src.core.observability import setup_observability

        probe_libraries("probe at import")
        setup_observability(token=None, environment="test", service_name="kp")
        probe_libraries("probe api")
        """
    )
    worker = _run(
        """
        from src.workers.__main__ import configure_process

        configure_process("extract")
        probe_libraries("probe worker")
        """
    )

    assert lines + worker == [
        "probe at import: -",
        "probe api: asyncpg httpx",
        "probe worker: asyncpg httpx",
    ]


def test_importing_the_worker_leaves_the_api_config_alone() -> None:
    # Process-global setup belongs to the entry point, never to an import: the API
    # imports the queue module (/health), tests import the runner.
    lines = _run(
        """
        from src.core.observability import setup_observability

        setup_observability(token=None, environment="test", service_name="kp")
        import src.workers.__main__
        import src.workers.runner

        probe("probe after import")
        import logfire
        print(f"probe service={logfire.DEFAULT_LOGFIRE_INSTANCE.config.service_name}")
        """
    )

    # The trust check alone can't see a reconfiguration: worker and API both distrust.
    assert lines == [
        "probe after import: recording=True adopted=False",
        "probe service=kp",
    ]


def test_a_worker_adopts_no_incoming_trace_context() -> None:
    # Nothing reaches a worker with a trace context of its own: an attempt continues
    # the one our API stored in the row (ADR-027), never one from outside.
    lines = _run(
        """
        from src.workers.__main__ import configure_process

        configure_process("embed")
        probe("probe worker", flags="01")
        """
    )

    assert lines == ["probe worker: recording=True adopted=False"]


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_each_worker_reports_its_own_service_name(phase: str) -> None:
    # One OTel service per deployable: the API and each phase's compose service.
    lines = _run(
        f"""
        import logfire
        from src.core.config import settings
        from src.workers.__main__ import configure_process

        configure_process("{phase}")
        name = logfire.DEFAULT_LOGFIRE_INSTANCE.config.service_name
        print(f"probe {{name == settings.SERVICE_NAME + '-worker-{phase}'}}")
        """
    )

    assert lines == ["probe True"]


def test_worker_logs_go_through_our_structlog_bridge() -> None:
    # Stdlib loggers (SQLAlchemy, asyncpg, httpx) reach structlog's formatter.
    lines = _run(
        """
        import logging
        from src.workers.__main__ import configure_process

        assert logging.getLogger().handlers == [], "configured at import"
        configure_process("extract")
        [handler] = logging.getLogger().handlers
        print(f"probe {type(handler.formatter).__name__}")
        """
    )

    assert lines == ["probe ProcessorFormatter"]


def test_traced_embedding_calls_carry_no_api_key() -> None:
    # Outbound HTTP is traced in both processes, and a span records the full URL.
    # A refused connection still ends a span, so no network is needed.
    lines = _run(
        """
        import asyncio
        import httpx
        import src.ai.embeddings as embeddings
        from src.core.observability import setup_observability

        setup_observability(token=None, environment="test", service_name="kp")
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
        embeddings._GEMINI_BASE = "http://127.0.0.1:9/v1beta/models"

        async def call():
            async with httpx.AsyncClient() as client:
                service = embeddings.EmbeddingService(
                    "SECRET-KEY", "gemini-embedding-001", 8, http_client=client
                )
                try:
                    await service.embed_query("q")
                except RuntimeError:
                    pass

        asyncio.run(call())
        spans = [s for s in exporter.get_finished_spans() if s.name == "POST"]
        leaked = any(
            "SECRET-KEY" in str(v)
            for s in spans
            for v in [*(s.attributes or {}).values(), s.name]
        )
        print(f"probe embedding: http_spans={len(spans)} key_in_trace={leaked}")
        """
    )

    assert lines == ["probe embedding: http_spans=1 key_in_trace=False"]
