"""The document pipeline's base task: how its tasks retry and how they fail.

The phase decides, not the exception type. A pure phase (parsing bytes already read,
chunking stored text) raises PermanentFailure: the same input gives the same result,
so it isn't retried. Everything else touches the outside world (file system,
database, broker, embedding API) and is retried by Celery's autoretry. When no attempt
remains, Celery calls on_failure, which marks the document FAILED with a reason.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

import structlog
from celery import Task

from src.domain.documents import FailureReason, Transition
from src.models.document import Document
from src.workers.database import get_sync_session
from src.workers.transitions import guarded_update

logger = structlog.get_logger(__name__)


class PermanentFailure(Exception):
    """A failure no retry can fix."""

    def __init__(self, reason: FailureReason) -> None:
        # Celery rebuilds the exception from a pickle, as cls(*args), before
        # on_failure sees it; one that can't be rebuilt arrives wrapped, reason lost.
        super().__init__(reason)
        self.reason = reason


class EmbeddingError(Exception):
    """The embedding provider failed: retried, then embedding_failed."""


class ParseKilled(Exception):
    """The PDF parse child was SIGKILLed, not by our timeout: the OOM killer under
    the extract container's memory cap. It kills the biggest process at that
    instant, which depends on what else is parsing, so one kill is no verdict on the
    file: retried, then extraction_too_large."""


def failure_reason(exc: BaseException) -> FailureReason:
    if isinstance(exc, PermanentFailure):
        return exc.reason
    if isinstance(exc, EmbeddingError):
        return FailureReason.EMBEDDING_FAILED
    if isinstance(exc, ParseKilled):
        return FailureReason.EXTRACTION_TOO_LARGE
    return FailureReason.INTERNAL_ERROR


class PipelineTask(Task):  # type: ignore[misc]
    """Base of the tasks that take a document from upload to INDEXED or FAILED.

    Each task takes the document id as its first argument and names, in ``fails``,
    the transition that fails the stage it owns. ``retry_backoff`` (the delay before
    the first retry; it doubles each time, with full jitter) is set per task.
    """

    fails: ClassVar[Transition]
    autoretry_for = (Exception,)
    dont_autoretry_for = (PermanentFailure,)
    # Expected failures: logged without a traceback.
    throws = (PermanentFailure,)
    max_retries = 3
    retry_jitter = True

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        document_id = str(args[0])
        reason = failure_reason(exc)
        try:
            with get_sync_session() as session:
                moved = session.execute(
                    guarded_update(uuid.UUID(document_id), self.fails)
                    .values(failure_reason=reason)
                    .returning(Document.id)
                ).first()
                session.commit()
        except Exception as write_exc:  # noqa: BLE001 — never mask the task's own failure
            # Not terminal: the document stays where it was until the reconciler.
            logger.error(
                "could not record the failure; document left as it was",
                document_id=document_id,
                failure_reason=reason,
                error_type=type(write_exc).__name__,
            )
            return
        if moved is None:
            logger.info(
                "failure not recorded: the document has moved past this stage",
                document_id=document_id,
                failure_reason=reason,
            )
            return
        logger.error(
            "document status FAILED",
            document_id=document_id,
            failure_reason=reason,
            error_type=type(exc).__name__,
        )
