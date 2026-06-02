from __future__ import annotations

import asyncio
import random
import time
import uuid

import logfire
import sqlalchemy as sa
import structlog
from celery import Task

from src.ai.chunking.factory import get_chunker
from src.ai.embeddings import EmbeddingService
from src.core.config import settings
from src.domain.documents import DocumentStatus
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.workers.celery_app import celery_app
from src.workers.database import get_sync_session

logger = structlog.get_logger(__name__)


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
    bind=True,
    max_retries=3,
    default_retry_delay=60,
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
                logger.warning(
                    "document has no raw_text, skipping", document_id=document_id
                )
                return

            raw_text = doc.raw_text
            content_type = doc.content_type
            doc_version = doc.version

        chunker = get_chunker(content_type)
        chunk_start = time.monotonic()
        with logfire.span(
            "chunk_text", document_id=document_id, content_type=content_type.value
        ) as chunk_span:
            chunk_data_list = chunker.chunk(raw_text, {"document_id": document_id})
            chunk_span.set_attribute("chunk_count", len(chunk_data_list))
            chunk_span.set_attribute(
                "chunking_duration_ms", round((time.monotonic() - chunk_start) * 1000)
            )

        if not chunk_data_list:
            logger.info("no chunks produced", document_id=document_id)
            return

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
                logger.error(
                    "embedding generation failed",
                    document_id=document_id,
                    error_type=type(exc).__name__,
                )
                if self.request.retries >= self.max_retries:
                    with get_sync_session() as fail_session:
                        fail_doc = fail_session.get(Document, doc_uuid)
                        if fail_doc is not None:
                            fail_doc.status = DocumentStatus.FAILED
                            fail_session.commit()
                    logger.error(
                        "embed_chunks exhausted all retries, marking FAILED",
                        document_id=document_id,
                    )
                raise self.retry(
                    exc=exc,
                    countdown=int(
                        2**self.request.retries * self.default_retry_delay
                        + random.uniform(0, self.default_retry_delay / 3)
                    ),
                ) from exc
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
                version=doc_version,
            )
            for i, (cd, embedding) in enumerate(
                zip(chunk_data_list, embeddings, strict=True)
            )
        ]

        # Re-indexing transaction with optimistic version check:
        # Lock the document row, verify its version hasn't been bumped by a concurrent
        # re-index while we were embedding (TOCTOU guard), then atomically delete stale
        # chunks (version < doc_version) and insert the new ones.
        with get_sync_session() as session:
            current_doc = session.execute(
                sa.select(Document).where(Document.id == doc_uuid).with_for_update()
            ).scalar_one_or_none()
            if current_doc is None or current_doc.version != doc_version:
                logger.info(
                    "doc version changed during embedding, discarding stale chunks",
                    document_id=document_id,
                    expected_version=doc_version,
                    actual_version=current_doc.version if current_doc else None,
                )
                return
            session.execute(
                sa.delete(DocumentChunk).where(
                    DocumentChunk.document_id == doc_uuid,
                    DocumentChunk.version < doc_version,
                )
            )
            session.add_all(new_chunks)
            current_doc.status = DocumentStatus.INDEXED
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
