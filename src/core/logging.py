import logging
import sys

import structlog
from opentelemetry import trace
from structlog.types import EventDict, WrappedLogger


def _inject_otel_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def setup_logging(*, environment: str = "development", log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _inject_otel_context,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor
    if environment == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # structlog-native loggers write directly via PrintLoggerFactory.
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib loggers (uvicorn, SQLAlchemy, asyncpg, Celery) through the
    # same processor chain so all log output shares trace_id/span_id and format.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,  # safe here — stdlib loggers have .name
            *shared_processors,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)
