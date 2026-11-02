from __future__ import annotations

import asyncio
import time
import uuid

import logfire
import sqlalchemy as sa
import structlog
from celery import Task
from sqlalchemy.orm import Session

from src.ai.chunking.factory import get_chunker
from src.ai.embeddings import EmbeddingService
from src.core.config import settings
from src.domain.documents import FAIL_EMBEDDING, DocumentStatus, FailureReason
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.workers.celery_app import celery_app
from src.workers.database import get_sync_session
from src.workers.pipeline import EmbeddingError, PermanentFailure, PipelineTask

logger = structlog.get_logger(__name__)


def _replace_chunks(
    session: Session, document_id: uuid.UUID, chunks: list[DocumentChunk]
) -> bool:
    """Swap in ``chunks`` as the document's whole chunk set and mark it INDEXED.

    Under the document's row lock, so a concurrent swap waits and then replaces this
    one's set instead of adding a second. The caller commits. False: the document is
    gone, nothing written.
    """
    doc = session.execute(
        sa.select(Document).where(Document.id == document_id).with_for_update()
    ).scalar_one_or_none()
    if doc is None:
        return False
    session.execute(
        sa.delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    session.add_all(chunks)
    doc.status = DocumentStatus.INDEXED
    # A success beats a rival's failure; its reason goes with it.
    doc.failure_reason = None
    return True


async def _run_embedding(
    texts: list[str],
    api_key: str,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    service = EmbeddingService(
        api_key=api_key,
        model=model,
        dimensions=dimensions,
    )
    return await service.embed_texts(texts)


@celery_app.task(  # type: ignore[untyped-decorator]
    base=PipelineTask,
    bind=True,
    fails=FAIL_EMBEDDING,
    retry_backoff=60,
    acks_late=True,
    soft_time_limit=settings.CELERY_EMBED_SOFT_TIME_LIMIT,
    time_limit=settings.CELERY_EMBED_TIME_LIMIT,
)
def embed_chunks(self: Task, document_id: str) -> None:
    with logfire.span("embed_chunks", document_id=document_id):
        doc_uuid = uuid.UUID(document_id)

        with get_sync_session() as session:
            doc = session.get(Document, doc_uuid)
            if doc is None:
                logger.warning("document not found, skipping", document_id=document_id)
                return
            if doc.status == DocumentStatus.INDEXED:
                logger.info("already indexed, skipping", document_id=document_id)
                return
            if doc.status != DocumentStatus.READY:
                logger.info(
                    "document not ready, skipping embedding",
                    document_id=document_id,
                    status=doc.status,
                )
                return
            if doc.raw_text is None:
                # Extraction writes READY and raw_text in one UPDATE: a bug.
                raise PermanentFailure(FailureReason.INTERNAL_ERROR)

            raw_text = doc.raw_text
            content_type = doc.content_type

        chunk_start = time.monotonic()
        with logfire.span(
            "chunk_text", document_id=document_id, content_type=content_type.value
        ) as chunk_span:
            # A pure function of the stored text (the soft limit included): retrying
            # can only fail again.
            try:
                chunker = get_chunker(content_type)
                chunk_data_list = chunker.chunk(raw_text, {"document_id": document_id})
            except Exception as exc:
                raise PermanentFailure(FailureReason.INTERNAL_ERROR) from exc
            chunk_span.set_attribute("chunk_count", len(chunk_data_list))
            chunk_span.set_attribute(
                "chunking_duration_ms", round((time.monotonic() - chunk_start) * 1000)
            )

        # The chunker is the one definition of searchable: a scanned PDF extracts to
        # page separators only, and whitespace yields no chunk.
        if not chunk_data_list:
            raise PermanentFailure(FailureReason.NO_TEXT)

        texts = [cd.text for cd in chunk_data_list]
        logger.info(
            "document status INDEXING",
            document_id=document_id,
            chunk_count=len(chunk_data_list),
        )

        with logfire.span("embed_batch", document_id=document_id) as embed_span:
            embed_start = time.monotonic()
            try:
                embeddings = asyncio.run(
                    _run_embedding(
                        texts=texts,
                        api_key=settings.GOOGLE_API_KEY,
                        model=settings.EMBEDDING_MODEL,
                        dimensions=settings.EMBEDDING_DIMENSIONS,
                    )
                )
            except Exception as exc:
                # Retried by autoretry; after the last attempt, embedding_failed.
                logger.error(
                    "embedding generation failed",
                    document_id=document_id,
                    error_type=type(exc).__name__,
                )
                raise EmbeddingError(type(exc).__name__) from exc
            embed_ms = round((time.monotonic() - embed_start) * 1000)
            embed_span.set_attribute("embedding_duration_ms", embed_ms)

        new_chunks = [
            DocumentChunk(
                document_id=doc_uuid,
                chunk_index=i,
                text=cd.text,
                embedding=embedding,
                token_count=cd.token_count,
                metadata_=cd.metadata,
            )
            for i, (cd, embedding) in enumerate(
                zip(chunk_data_list, embeddings, strict=True)
            )
        ]

        # Another run of this document (a redelivered or requeued message) may have
        # swapped in meanwhile; it chunked the same raw_text, so replacing its set
        # changes nothing a reader sees.
        with get_sync_session() as session:
            if not _replace_chunks(session, doc_uuid, new_chunks):
                logger.info(
                    "document deleted during embedding, discarding chunks",
                    document_id=document_id,
                )
                return
            session.commit()

        total_tokens = sum(cd.token_count for cd in chunk_data_list)
        logger.info(
            "embedding complete",
            document_id=document_id,
            status="INDEXED",
            chunk_count=len(new_chunks),
            total_tokens=total_tokens,
            embedding_duration_ms=embed_ms,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
        )
