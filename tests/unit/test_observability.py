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

ROOT = Path(__file__).resolve().parents[2]

# An empty token beats .env's (pydantic-settings reads the environment first), so
# the worker scenario, which configures Logfire from settings, never exports.
_OFFLINE_ENV = {**os.environ, "LOGFIRE_TOKEN": ""}

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
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    instrumentors = {
        "asyncpg": AsyncPGInstrumentor(),
        "celery": CeleryInstrumentor(),
        "httpx": HTTPXClientInstrumentor(),
    }
    traced = [n for n, i in instrumentors.items() if i.is_instrumented_by_opentelemetry]
    print(f"{label}: {' '.join(traced) or '-'}")
"""


def _run(scenario: str) -> list[str]:
    code = _PROBE + textwrap.dedent(scenario)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_OFFLINE_ENV,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith("probe ")]


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
    # Database queries, task enqueueing/execution and outbound HTTP, in both
    # processes; nothing before setup (importing configures nothing).
    lines = _run(
        """
        from celery.signals import worker_process_init
        import src.main
        import src.workers.celery_app
        from src.core.observability import setup_observability

        probe_libraries("probe at import")
        setup_observability(token=None, environment="test", service_name="kp")
        probe_libraries("probe api")
        """
    )
    worker = _run(
        """
        from celery.signals import worker_process_init
        import src.workers.celery_app

        worker_process_init.send(sender=None)
        probe_libraries("probe worker")
        """
    )

    assert lines + worker == [
        "probe at import: -",
        "probe api: asyncpg celery httpx",
        "probe worker: asyncpg celery httpx",
    ]


def test_importing_the_celery_app_leaves_the_api_config_alone() -> None:
    # The API imports it lazily (health check, document upload); an import-time
    # logfire.configure() there used to reset the API's settings.
    lines = _run(
        """
        from src.core.observability import setup_observability

        setup_observability(token=None, environment="test", service_name="kp")
        import src.workers.celery_app

        probe("probe after import")
        """
    )

    assert lines == ["probe after import: recording=True adopted=False"]


def test_worker_process_trusts_trace_context_from_our_api() -> None:
    # Task context arrives only via our broker, injected by our API, so a worker
    # joins the API's trace. Configured per process, after the prefork pool forks:
    # nothing records before the signal.
    lines = _run(
        """
        from celery.signals import worker_process_init
        import src.workers.celery_app

        probe("probe before", flags="01")
        worker_process_init.send(sender=None)
        probe("probe worker", flags="01")
        """
    )

    assert lines == [
        "probe before: recording=False adopted=True",
        "probe worker: recording=True adopted=True",
    ]


def test_worker_logging_is_ours_not_celerys() -> None:
    # Without a setup_logging receiver Celery clears the root logger's handlers,
    # discarding our structlog bridge; with one, it leaves logging to us.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import logging
                from celery.signals import setup_logging
                import src.workers.celery_app

                assert logging.getLogger().handlers == [], "configured at import"
                responses = setup_logging.send(
                    sender=None,
                    loglevel="INFO",
                    logfile=None,
                    format="",
                    colorize=False,
                )
                [handler] = logging.getLogger().handlers
                print(len(responses), type(handler.formatter).__name__)
                """
            ),
        ],
        cwd=ROOT,
        env=_OFFLINE_ENV,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1", "ProcessorFormatter"]
