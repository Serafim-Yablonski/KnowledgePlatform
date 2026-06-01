from typing import Any

import logfire
from celery import Celery
from celery.signals import worker_shutdown

from src.core.config import settings
from src.core.redis import close_sync_redis

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
)

# Configure logfire for Celery worker processes.
# FastAPI processes do this via setup_observability(); calling twice is safe.
if settings.LOGFIRE_TOKEN:
    logfire.configure(token=settings.LOGFIRE_TOKEN)
else:
    logfire.configure(send_to_logfire=False)
logfire.instrument_celery()


# Problem 7 fix: release the lazily-created sync Redis client when the worker
# exits so connections are closed cleanly rather than abandoned.
@worker_shutdown.connect  # type: ignore[untyped-decorator]
def _on_worker_shutdown(**kwargs: Any) -> None:
    close_sync_redis()
