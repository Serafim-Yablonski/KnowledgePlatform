from __future__ import annotations

import asyncio
import time
import uuid

import logfire
import redis.asyncio as aioredis
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
    redis_url: str,
) -> list[list[float]]:
    """Run async embedding in asyncio.run(). Creates and closes its own Redis client."""
    redis_client = aioredis.Redis.from_url(redis_url, decode_responses=True)
    try:
        service = EmbeddingService(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            redis_client=redis_client,
        )
        return await service.embed_texts(texts)
    finally:
        await redis_client.aclose()


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def embed_chunks(self: Task, document_id: str) -> None:
    with logfire.span("embed_chunks", document_id=document_id):
        doc_uuid = uuid.UUID(document_id)

        with get_sync_session() as session:
            doc = session.get(Document, doc_uuid)
            if doc is None:
                logger.warning("document not found, skipping", document_id=document_id)
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
        chunk_data_list = chunker.chunk(raw_text, {"document_id": document_id})

        if not chunk_data_list:
            logger.info("no chunks produced", document_id=document_id)
            return

        texts = [cd.text for cd in chunk_data_list]

        embed_start = time.monotonic()
        try:
            embeddings = asyncio.run(
                _run_embedding(
                    texts=texts,
                    api_key=settings.GOOGLE_API_KEY,
                    model=settings.EMBEDDING_MODEL,
                    dimensions=settings.EMBEDDING_DIMENSIONS,
                    redis_url=settings.REDIS_URL,
                )
            )
        except Exception as exc:
            logger.error(
                "embedding generation failed",
                document_id=document_id,
                error=str(exc),
            )
            raise self.retry(exc=exc) from exc
        embed_ms = round((time.monotonic() - embed_start) * 1000)

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

        # Re-indexing transaction:
        # DELETE chunks with version < doc_version, then INSERT the new ones.
        # Using version < (not !=) means a concurrent worker running an older retry
        # will never delete chunks inserted by a newer run, preventing races.
        with get_sync_session() as session:
            session.execute(
                sa.delete(DocumentChunk).where(
                    DocumentChunk.document_id == doc_uuid,
                    DocumentChunk.version < doc_version,
                )
            )
            session.add_all(new_chunks)
            session.commit()

        total_tokens = sum(cd.token_count for cd in chunk_data_list)
        logger.info(
            "embedding complete",
            document_id=document_id,
            chunk_count=len(new_chunks),
            total_tokens=total_tokens,
            embedding_duration_ms=embed_ms,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
        )
