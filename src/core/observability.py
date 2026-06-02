import logfire
import structlog
from fastapi import FastAPI

_logger = structlog.get_logger(__name__)


def setup_observability(
    app: FastAPI,
    *,
    token: str | None = None,
    environment: str = "development",
    service_name: str = "knowledge-platform",
) -> None:
    if token:
        logfire.configure(
            token=token,
            service_name=service_name,
            environment=environment,
        )
    else:
        logfire.configure(
            send_to_logfire=False,
            service_name=service_name,
            environment=environment,
        )
    logfire.instrument_fastapi(app)
    # asyncpg, celery, and httpx instrumentation require optional OTel packages
    # that may not be available for all Python versions — warn explicitly when missing.
    try:
        logfire.instrument_asyncpg()
    except RuntimeError as exc:
        _logger.warning("asyncpg instrumentation unavailable", reason=str(exc))
    try:
        logfire.instrument_celery()
    except RuntimeError as exc:
        _logger.warning("celery instrumentation unavailable", reason=str(exc))
    try:
        logfire.instrument_httpx()
    except RuntimeError as exc:
        _logger.warning("httpx instrumentation unavailable", reason=str(exc))


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
