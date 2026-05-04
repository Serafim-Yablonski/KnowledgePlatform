import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import logfire
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

P = ParamSpec("P")
T = TypeVar("T")


def setup_observability(app: FastAPI, *, token: str | None = None) -> None:
    if token:
        logfire.configure(token=token)
    else:
        logfire.configure(send_to_logfire=False)
    logfire.instrument_fastapi(app)
    logfire.instrument_asyncpg()
    logfire.instrument_celery()


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
