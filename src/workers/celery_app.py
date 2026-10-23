from typing import Any

from celery import Celery, signals

from src.core.config import settings
from src.core.logging import setup_logging
from src.core.observability import configure_logfire, instrument_libraries
from src.core.redis import close_sync_redis

# Structural constants — queue names do not vary by environment.
QUEUE_EXTRACT = "extract"
QUEUE_EMBED = "embed"

celery_app = Celery(
    "knowledge_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    # Explicit False: eager mode masks broker failures; never use it, even in tests.
    task_always_eager=False,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # extract_text returns None and no caller reads the result; skip storing it.
    task_ignore_result=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Restart workers after N tasks to prevent memory creep from asyncio.run() loops.
    worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
    # Retry connecting to Redis at startup rather than hard-failing (Celery 5.3+).
    broker_connection_retry_on_startup=True,
    # Emit STARTED/SUCCESS/FAILURE/RETRY events so celery inspect can show which
    # tasks are currently executing.
    worker_send_task_events=True,
    # Route CPU-bound extraction and I/O-bound embedding to separate queues so
    # each pool can be scaled independently.
    task_routes={
        "src.workers.tasks.extract_text.extract_text": {"queue": QUEUE_EXTRACT},
        "src.workers.tasks.embed_chunks.embed_chunks": {"queue": QUEUE_EMBED},
    },
)

# Worker setup runs from Celery signals, never at import: the API process imports
# this module lazily (health check, task enqueueing), and process-global setup
# here would overwrite the API's own logging and tracing config.


# Connecting this signal is how Celery hands logging over: without a receiver it
# clears the root logger's handlers (worker_hijack_root_logger), dropping ours.
@signals.setup_logging.connect(weak=False)  # type: ignore[untyped-decorator]
def _configure_worker_logging(**kwargs: Any) -> None:
    setup_logging(environment=settings.ENVIRONMENT, log_level=settings.LOG_LEVEL)


# Per process, after the prefork pool forks (and again for each replacement child
# under worker_max_tasks_per_child): Logfire's exporter thread wouldn't survive
# being created in the parent before the fork. Only the prefork (compose's default)
# and solo pools send this signal: under threads/gevent/eventlet tracing stays off.
@signals.worker_process_init.connect(weak=False)  # type: ignore[untyped-decorator]
def _configure_worker_tracing(**kwargs: Any) -> None:
    # Task trace context comes only from our API via the broker: trusted, so a
    # task's spans join the request that enqueued it.
    configure_logfire(
        token=settings.LOGFIRE_TOKEN or None,
        environment=settings.ENVIRONMENT,
        service_name=settings.SERVICE_NAME,
        trust_incoming_trace_context=True,
    )
    instrument_libraries()


# Problem 7 fix: release the lazily-created sync Redis client when the worker
# exits so connections are closed cleanly rather than abandoned.
@signals.worker_shutdown.connect  # type: ignore[untyped-decorator]
def _on_worker_shutdown(**kwargs: Any) -> None:
    close_sync_redis()
