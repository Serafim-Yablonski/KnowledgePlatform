from __future__ import annotations

import random
import time
import uuid
from pathlib import Path

import logfire
import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from pypdf.errors import PdfReadError

from src.core.config import settings
from src.domain.documents import ContentType, DocumentStatus
from src.models.document import Document
from src.workers.celery_app import celery_app
from src.workers.database import get_sync_session

logger = structlog.get_logger(__name__)


def _read_text(file_path: str, content_type: ContentType) -> str:
    if content_type == ContentType.PDF:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return Path(file_path).read_text(encoding="utf-8")


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    soft_time_limit=settings.CELERY_EXTRACT_SOFT_TIME_LIMIT,
    time_limit=settings.CELERY_EXTRACT_TIME_LIMIT,
)
def extract_text(self: Task, document_id: str) -> None:
    with logfire.span("extract_text", document_id=document_id) as span:
        start_ms = time.monotonic() * 1000
        doc_uuid = uuid.UUID(document_id)

        with get_sync_session() as session:
            doc = session.get(Document, doc_uuid)
            if doc is None:
                logger.warning("document not found, skipping", document_id=document_id)
                return
            # Skip if already done or fully indexed. PROCESSING is intentionally
            # not checked: a crashed worker must be able to retry. INDEXED is
            # checked to prevent status regression on duplicate messages.
            if doc.status in (DocumentStatus.READY, DocumentStatus.INDEXED):
                return

            file_path = doc.file_path
            content_type = doc.content_type
            file_size = doc.file_size_bytes

            doc.status = DocumentStatus.PROCESSING
            session.commit()
            logger.info(
                "document status PROCESSING",
                document_id=document_id,
                content_type=content_type.value,
            )

            # Initialize before the try so the success log below is always bound.
            raw_text = ""
            extraction_succeeded = False
            try:
                raw_text = _read_text(file_path, content_type)
                doc.raw_text = raw_text
                doc.status = DocumentStatus.READY
                session.commit()
                extraction_succeeded = True
                logger.info(
                    "document status READY",
                    document_id=document_id,
                    extracted_text_length=len(raw_text),
                )
            except (
                OSError,
                UnicodeDecodeError,
                PdfReadError,
                SoftTimeLimitExceeded,
            ) as exc:
                logger.error(
                    "text extraction failed",
                    document_id=document_id,
                    status="FAILED",
                    error_type=type(exc).__name__,
                )
                doc.status = DocumentStatus.FAILED
                try:
                    session.commit()
                except Exception as commit_exc:
                    logger.warning(
                        "failed to persist FAILED status",
                        document_id=document_id,
                        error_type=type(commit_exc).__name__,
                    )
                raise self.retry(
                    exc=exc,
                    countdown=int(
                        2**self.request.retries * self.default_retry_delay
                        + random.uniform(0, self.default_retry_delay / 3)
                    ),
                ) from exc

        duration_ms = time.monotonic() * 1000 - start_ms
        span.set_attribute("extraction_duration_ms", round(duration_ms))
        logger.info(
            "text extraction complete",
            document_id=document_id,
            file_size_bytes=file_size,
            extracted_text_length=len(raw_text),
            extraction_duration_ms=round(duration_ms),
        )

        # Dispatch AFTER the session is fully closed so the commit is durable before
        # the embed worker reads the document. Placing this inside the session block
        # risks enqueuing the task before the connection is released (outbox pattern).
        if extraction_succeeded:
            from src.workers.tasks.embed_chunks import embed_chunks

            embed_chunks.delay(document_id)
