from __future__ import annotations

import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import logfire
import sqlalchemy as sa
import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from src.core.config import settings
from src.domain.documents import (
    FAIL_EXTRACTION,
    FINISH_EXTRACTION,
    START_EXTRACTION,
    ContentType,
    DocumentStatus,
    FailureReason,
)
from src.models.document import Document
from src.workers.celery_app import celery_app
from src.workers.database import get_sync_session
from src.workers.pipeline import ParseKilled, PermanentFailure, PipelineTask
from src.workers.transitions import guarded_update

logger = structlog.get_logger(__name__)


# Run by path, not imported: see the module's docstring.
_PDF_CHILD = Path(__file__).resolve().parents[1] / "pdf_text.py"


def _read_file(file_path: str) -> bytes:
    return Path(file_path).read_bytes()


def _parse_pdf(data: bytes) -> str:
    """The PDF's text, parsed in a child process; its exit says whose fault a failure
    is. The child only gets bytes: file-system errors stay in the retried read."""
    from src.workers.pdf_text import PARSE_FAILED

    try:
        done = subprocess.run(
            [sys.executable, "-I", str(_PDF_CHILD)],
            input=data,
            capture_output=True,
            timeout=settings.CELERY_EXTRACT_PARSE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # subprocess.run has killed the child
        raise PermanentFailure(FailureReason.EXTRACTION_TIMEOUT) from exc
    if done.returncode == 0:
        return done.stdout.decode("utf-8")
    if done.returncode == -signal.SIGKILL:
        # Not our timeout (above): the OOM killer under the container's memory cap.
        # Its victim depends on what else is parsing: an attempt, not a verdict.
        logger.warning("PDF parse killed (out of memory?)", exit_code=done.returncode)
        raise ParseKilled(f"PDF parser killed by signal {signal.SIGKILL.name}")
    # PARSE_FAILED: pypdf raised. Another signal: a crash (segfault, abort), which the
    # same bytes repeat. Both are the file's.
    if done.returncode == PARSE_FAILED or done.returncode < 0:
        logger.info(
            "PDF parse failed",
            exit_code=done.returncode,
            parser_error=done.stderr[:100].decode("utf-8", "replace"),
        )
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)
    # Anything else is ours (e.g. 1: an import failed): retried, then internal_error.
    raise RuntimeError(
        f"PDF parser exited with {done.returncode}: "
        f"{done.stderr[-500:].decode('utf-8', 'replace')}"
    )


def _parse(data: bytes, content_type: ContentType) -> str:
    if content_type == ContentType.PDF:
        return _parse_pdf(data)
    return data.decode("utf-8")


def _extract(data: bytes, content_type: ContentType) -> str:
    """The file's text; failures the file causes are permanent."""
    try:
        return _parse(data, content_type)
    except SoftTimeLimitExceeded as exc:
        # A backstop: the child's own timeout comes before the soft limit.
        raise PermanentFailure(FailureReason.EXTRACTION_TIMEOUT) from exc
    except UnicodeDecodeError as exc:
        raise PermanentFailure(FailureReason.UNREADABLE_FILE) from exc


def _dispatch_embedding(document_id: str) -> None:
    """Hand the document to embed_chunks. Only after READY is committed, so the embed
    worker reads it; READY records that this is owed, so a re-run sends it again."""
    from src.workers.tasks.embed_chunks import embed_chunks

    embed_chunks.delay(document_id)


@celery_app.task(  # type: ignore[untyped-decorator]
    base=PipelineTask,
    bind=True,
    fails=FAIL_EXTRACTION,
    retry_backoff=30,
    acks_late=True,
    soft_time_limit=settings.CELERY_EXTRACT_SOFT_TIME_LIMIT,
    time_limit=settings.CELERY_EXTRACT_TIME_LIMIT,
)
def extract_text(self: Task, document_id: str) -> None:
    with logfire.span("extract_text", document_id=document_id) as span:
        start_ms = time.monotonic() * 1000
        doc_uuid = uuid.UUID(document_id)

        with get_sync_session() as session:
            claimed = session.execute(
                guarded_update(doc_uuid, START_EXTRACTION).returning(
                    Document.file_path,
                    Document.content_type,
                    Document.file_size_bytes,
                )
            ).first()
            session.commit()
            # Only a miss pays this read. READY: extracted, but this message runs
            # again (the worker was lost before the dispatch, or the dispatch failed
            # and this is its retry), so the hand-off is still owed.
            handoff_owed = (
                claimed is None
                and session.scalar(
                    sa.select(Document.status).where(Document.id == doc_uuid)
                )
                == DocumentStatus.READY
            )
        if handoff_owed:
            logger.info(
                "already extracted; dispatching embedding again",
                document_id=document_id,
            )
            _dispatch_embedding(document_id)
            return
        if claimed is None:
            logger.info(
                "document gone, past extraction or failed; skipping",
                document_id=document_id,
            )
            return
        file_path, content_type, file_size = claimed
        logger.info(
            "document status PROCESSING",
            document_id=document_id,
            content_type=content_type.value,
        )

        # Retried on OSError; if the document was deleted meanwhile, the next
        # attempt's claim finds nothing and stops.
        data = _read_file(file_path)
        raw_text = _extract(data, content_type)

        with get_sync_session() as session:
            saved = session.execute(
                guarded_update(doc_uuid, FINISH_EXTRACTION)
                .values(raw_text=raw_text)
                .returning(Document.id)
            ).first()
            session.commit()
        if saved is None:
            # A rival run moved it on (or it was deleted): its outcome stands.
            logger.info(
                "document moved on during extraction, discarding text",
                document_id=document_id,
            )
            return

        duration_ms = time.monotonic() * 1000 - start_ms
        span.set_attribute("extraction_duration_ms", round(duration_ms))
        logger.info(
            "document status READY",
            document_id=document_id,
            file_size_bytes=file_size,
            extracted_text_length=len(raw_text),
            extraction_duration_ms=round(duration_ms),
        )

    _dispatch_embedding(document_id)
