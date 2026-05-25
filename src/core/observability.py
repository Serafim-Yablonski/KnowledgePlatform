import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import logfire
import structlog
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

P = ParamSpec("P")
T = TypeVar("T")

_logger = structlog.get_logger(__name__)


def setup_observability(
    app: FastAPI,
    *,
    token: str | None = None,
    environment: str = "development",
) -> None:
    if token:
        logfire.configure(
            token=token,
            service_name="knowledge-platform",
            environment=environment,
        )
    else:
        logfire.configure(
            send_to_logfire=False,
            service_name="knowledge-platform",
            environment=environment,
        )
    logfire.instrument_fastapi(app)
    # asyncpg and celery instrumentation require optional OTel packages that may
    # not be available for all Python versions — warn explicitly when missing.
    try:
        logfire.instrument_asyncpg()
    except RuntimeError as exc:
        _logger.warning("asyncpg instrumentation unavailable", reason=str(exc))
    try:
        logfire.instrument_celery()
    except RuntimeError as exc:
        _logger.warning("celery instrumentation unavailable", reason=str(exc))


def traced(
    span_name: str | None = None,
    attributes: Attributes = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        name = span_name or fn.__qualname__
        tracer = trace.get_tracer(__name__)

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(name, attributes=attributes):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


def log_span_event(
    span: Span, event_name: str, attributes: dict[str, Any] | None = None
) -> None:
    span.add_event(event_name, attributes=attributes or {})


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
