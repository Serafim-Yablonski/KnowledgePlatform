"""The pipeline's two jobs, run once a worker's claim has committed (ADR-027).

Each reads its input in a short transaction, works outside any transaction and writes
its result in one guarded transaction. A job ends one of three ways, and the worker
maps each to an outcome: it returns (done, or nothing left to do); it raises
PermanentFailure (the phase decides: the same input would fail again); or it raises
anything else, which is transient and retried. ParseKilled and EmbeddingError name
the reason a used-up budget records (``failure_reason``).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Protocol

import logfire
import sqlalchemy as sa
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.ai.chunking.factory import get_chunker
from src.domain.documents import (
    FINISH_EXTRACTION,
    ContentType,
    DocumentStatus,
    FailureReason,
)
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.workers.pipeline import EmbeddingError, ParseKilled, PermanentFailure
from src.workers.queue import hand_off
from src.workers.transitions import guarded_update

logger = structlog.get_logger(__name__)

# Run by path, not imported: see pdf_text's docstring.
_PDF_CHILD = Path(__file__).resolve().parent / "pdf_text.py"


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


async def _parse_pdf(data: bytes, timeout: float) -> str:
    """The PDF's text, parsed in a child process; its exit says whose fault a failure
    is. The child never outlives the call: a timeout or a cancellation kills it."""
    from src.workers.pdf_text import PARSE_FAILED

    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(_PDF_CHILD),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(data), timeout)
    except TimeoutError as exc:
        raise PermanentFailure(FailureReason.EXTRACTION_TIMEOUT) from exc
    finally:
        if child.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                child.kill()
            await child.wait()
    # The child has exited (communicate waits for it); wait() returns its code.
    code = await child.wait()
    if code == 0:
        return stdout.decode("utf-8")
    if code == -signal.SIGKILL:
        # Not our timeout (above): the OOM killer under the container's memory cap.
        # Its victim depends on what else is parsing: an attempt, not a verdict.
        logger.warning("PDF parse killed (out of memory?)", exit_code=code)
        raise ParseKilled(f"PDF parser killed by signal {signal.SIGKILL.name}")
    # PARSE_FAILED: pypdf raised. Another signal: a crash (segfault, abort), which the
    # same bytes repeat. Both are the file's.
    if code == PARSE_FAILED or code < 0:
        logger.info(
            "PDF parse failed",
            exit_code=code,
            parser_error=stderr[:100].decode("utf-8", "replace"),
        )
        raise PermanentFailure(FailureReason.UNREADABLE_FILE)
    # Anything else is ours (e.g. 1: an import failed): transient.
    raise RuntimeError(
        f"PDF parser exited with {code}: {stderr[-500:].decode('utf-8', 'replace')}"
    )


async def _extract(data: bytes, content_type: ContentType, parse_timeout: float) -> str:
    """The file's text; failures the file causes are permanent."""
    if content_type == ContentType.PDF:
        return await _parse_pdf(data, parse_timeout)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PermanentFailure(FailureReason.UNREADABLE_FILE) from exc


async def extract_text(
    sessions: async_sessionmaker[AsyncSession],
    document_id: uuid.UUID,
    *,
    parse_timeout: float,
) -> None:
    """Extract a claimed document's text and hand it to embedding (READY)."""
    with logfire.span("extract_text", document_id=str(document_id)) as span:
        start = time.monotonic()
        async with sessions() as session:
            row = (
                await session.execute(
                    sa.select(
                        Document.file_path,
                        Document.content_type,
                        Document.file_size_bytes,
                    ).where(Document.id == document_id)
                )
            ).first()
        if row is None:
            logger.info("document gone; skipping", document_id=str(document_id))
            return
        file_path, content_type, file_size = row

        # An OSError is transient; a document deleted meanwhile is found gone by the
        # next attempt's claim.
        data = await asyncio.to_thread(Path(file_path).read_bytes)
        raw_text = await _extract(data, content_type, parse_timeout)

        async with sessions.begin() as session:
            saved = (
                await session.execute(
                    guarded_update(document_id, FINISH_EXTRACTION)
                    .values(raw_text=raw_text, **hand_off())
                    .returning(Document.id)
                )
            ).first()
        if saved is None:
            # A rival run moved it on (or it was deleted): its outcome stands.
            logger.info(
                "document moved on during extraction, discarding text",
                document_id=str(document_id),
            )
            return

        duration_ms = round((time.monotonic() - start) * 1000)
        span.set_attribute("extraction_duration_ms", duration_ms)
        logger.info(
            "document status READY",
            document_id=str(document_id),
            file_size_bytes=file_size,
            extracted_text_length=len(raw_text),
            extraction_duration_ms=duration_ms,
        )


async def replace_chunks(
    session: AsyncSession, document_id: uuid.UUID, chunks: list[DocumentChunk]
) -> bool:
    """Swap in ``chunks`` as the document's whole chunk set and mark it INDEXED.

    Under the document's row lock, so a concurrent swap waits and then replaces this
    one's set instead of adding a second. The caller commits. False: the document is
    gone, nothing written.
    """
    doc = (
        await session.execute(
            sa.select(Document).where(Document.id == document_id).with_for_update()
        )
    ).scalar_one_or_none()
    if doc is None:
        return False
    await session.execute(
        sa.delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    session.add_all(chunks)
    doc.status = DocumentStatus.INDEXED
    # A success beats a rival's failure; its reason goes with it.
    doc.failure_reason = None
    return True


async def embed_chunks(
    sessions: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    document_id: uuid.UUID,
) -> None:
    """Chunk and embed a claimed document's text, then swap in its chunk set."""
    with logfire.span("embed_chunks", document_id=str(document_id)):
        async with sessions() as session:
            row = (
                await session.execute(
                    sa.select(
                        Document.status, Document.raw_text, Document.content_type
                    ).where(Document.id == document_id)
                )
            ).first()
        if row is None:
            logger.info("document gone; skipping", document_id=str(document_id))
            return
        status, raw_text, content_type = row
        if status != DocumentStatus.READY:
            # Claimed READY, moved on since: a rival run finished it, or it failed.
            logger.info(
                "document no longer READY; skipping",
                document_id=str(document_id),
                status=status,
            )
            return
        if raw_text is None:
            # Extraction writes READY and raw_text in one UPDATE: a bug.
            raise PermanentFailure(FailureReason.INTERNAL_ERROR)

        chunk_start = time.monotonic()
        with logfire.span(
            "chunk_text", document_id=str(document_id), content_type=content_type.value
        ) as chunk_span:
            # A pure function of the stored text: retrying can only fail again.
            try:
                chunker = get_chunker(content_type)
                chunk_data = await asyncio.to_thread(
                    chunker.chunk, raw_text, {"document_id": str(document_id)}
                )
            except Exception as exc:
                raise PermanentFailure(FailureReason.INTERNAL_ERROR) from exc
            chunk_span.set_attribute("chunk_count", len(chunk_data))
            chunk_span.set_attribute(
                "chunking_duration_ms", round((time.monotonic() - chunk_start) * 1000)
            )
        # The chunker is the one definition of searchable: a scanned PDF extracts to
        # page separators only, and whitespace yields no chunk.
        if not chunk_data:
            raise PermanentFailure(FailureReason.NO_TEXT)

        with logfire.span("embed_batch", document_id=str(document_id)) as embed_span:
            embed_start = time.monotonic()
            try:
                embeddings = await embedder.embed_texts([c.text for c in chunk_data])
            except Exception as exc:
                logger.error(
                    "embedding generation failed",
                    document_id=str(document_id),
                    error_type=type(exc).__name__,
                )
                raise EmbeddingError(type(exc).__name__) from exc
            embed_ms = round((time.monotonic() - embed_start) * 1000)
            embed_span.set_attribute("embedding_duration_ms", embed_ms)

        chunks = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                text=c.text,
                embedding=embedding,
                token_count=c.token_count,
                metadata_=c.metadata,
            )
            for i, (c, embedding) in enumerate(zip(chunk_data, embeddings, strict=True))
        ]
        # A rival run of this document (its lease ran out while this one worked) may
        # have swapped in meanwhile; it chunked the same raw_text, so replacing its
        # set changes nothing a reader sees.
        async with sessions.begin() as session:
            if not await replace_chunks(session, document_id, chunks):
                logger.info(
                    "document deleted during embedding, discarding chunks",
                    document_id=str(document_id),
                )
                return

        logger.info(
            "embedding complete",
            document_id=str(document_id),
            status="INDEXED",
            chunk_count=len(chunks),
            total_tokens=sum(c.token_count for c in chunk_data),
            embedding_duration_ms=embed_ms,
        )
