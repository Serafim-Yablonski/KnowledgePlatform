import contextlib
from typing import Any

import logfire
from celery import Celery
from celery.signals import worker_shutdown

from src.core.config import settings
from src.core.logging import setup_logging
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

# Configure logfire for Celery worker processes.
# FastAPI processes do this via setup_observability(); calling twice is safe.
if settings.LOGFIRE_TOKEN:
    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        service_name=settings.SERVICE_NAME,
        environment=settings.ENVIRONMENT,
    )
else:
    logfire.configure(
        send_to_logfire=False,
        service_name=settings.SERVICE_NAME,
        environment=settings.ENVIRONMENT,
    )
logfire.instrument_celery()
with contextlib.suppress(RuntimeError):
    logfire.instrument_httpx()
setup_logging(environment=settings.ENVIRONMENT, log_level=settings.LOG_LEVEL)


# Problem 7 fix: release the lazily-created sync Redis client when the worker
# exits so connections are closed cleanly rather than abandoned.
@worker_shutdown.connect  # type: ignore[untyped-decorator]
def _on_worker_shutdown(**kwargs: Any) -> None:
    close_sync_redis()
