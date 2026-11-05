"""How a pipeline job's failure is classified: whether a retry can fix it, and the
reason a document records when it can't (ADR-025, ADR-027).

The phase decides, not the exception type. A pure phase (parsing bytes already read,
chunking stored text) raises PermanentFailure: the same input gives the same result,
so it isn't retried. Everything else touches the outside world (file system,
database, embedding API) and is transient: the worker retries it after a backoff
until the phase's budget is used up, then records ``failure_reason``.
"""

from __future__ import annotations

from src.domain.documents import FailureReason


class PermanentFailure(Exception):
    """A failure no retry can fix."""

    def __init__(self, reason: FailureReason) -> None:
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
