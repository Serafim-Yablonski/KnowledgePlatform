from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import logfire

if TYPE_CHECKING:
    from fastapi import FastAPI


def configure_logfire(
    *,
    token: str | None,
    environment: str,
    service_name: str,
    trust_incoming_trace_context: bool,
) -> None:
    """Configure Logfire for this process; the one place both processes call.

    ``trust_incoming_trace_context`` decides whether a span may join a trace context
    the caller sent (an HTTP ``traceparent``, or MCP's ``_meta`` — both read through
    the global propagator Logfire installs). Only trust a caller we control: the
    default sampler follows the parent, so an untrusted "don't sample" flag would
    switch our recording off, and its trace id would become ours. Logfire's default
    (None) trusts it and warns once.
    """
    if token:
        logfire.configure(
            token=token,
            service_name=service_name,
            environment=environment,
            distributed_tracing=trust_incoming_trace_context,
        )
    else:
        logfire.configure(
            send_to_logfire=False,
            service_name=service_name,
            environment=environment,
            distributed_tracing=trust_incoming_trace_context,
        )


def instrument_app(app: FastAPI) -> None:
    """Trace the app's requests; call while building it, before it serves anything.

    Starlette builds its middleware stack on the first ASGI call, which is the
    lifespan's own, so instrumentation added during startup would never run. Spans
    start recording once ``setup_observability`` configures Logfire at startup.
    """
    with warnings.catch_warnings():
        # Logfire warns that instrumenting before configure() has no effect *yet*.
        warnings.filterwarnings("ignore", message="Instrumentation will have no effect")
        logfire.instrument_fastapi(app)


def instrument_libraries() -> None:
    """Trace database queries, Celery tasks and outbound HTTP; both processes."""
    logfire.instrument_asyncpg()
    logfire.instrument_celery()
    logfire.instrument_httpx()


def setup_observability(
    *,
    token: str | None,
    environment: str,
    service_name: str,
) -> None:
    """The API process's tracing config; run at startup, never at import."""
    # REST and MCP callers are untrusted: their trace context is ignored.
    configure_logfire(
        token=token,
        environment=environment,
        service_name=service_name,
        trust_incoming_trace_context=False,
    )
    instrument_libraries()


def set_llm_span_attrs(
    span: logfire.LogfireSpan,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Set standard OTel GenAI semantic attributes on a logfire span.

    Logfire reads gen_ai.* keys into dedicated columns and uses them to compute
    operation_cost automatically for known model names.
    """
    provider = model.split(":")[0].split("-")[0]
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
