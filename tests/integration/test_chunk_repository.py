"""Integration tests for SQLAlchemyChunkRepository against real PostgreSQL 18."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import ChunkText, ContentType, DocumentStatus
from src.models.chunk import EMBEDDING_DIMS, DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.repositories.chunk import SQLAlchemyChunkRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository

_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)


@pytest.fixture
async def document(db_session: AsyncSession, test_user: User) -> Document:
    ws = await SQLAlchemyWorkspaceRepository(db_session).create(
        name="Chunks", slug=f"chunks-{uuid.uuid4().hex[:8]}", created_by_id=test_user.id
    )
    doc = Document(
        workspace_id=ws.id,
        title="Runbook",
        content_type=ContentType.MARKDOWN,
        file_path="/tmp/runbook.md",
        file_size_bytes=100,
        uploaded_by=test_user.id,
        status=DocumentStatus.INDEXED,
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


def _chunk(
    doc: Document, index: int, *, metadata: dict[str, str] | None = None
) -> DocumentChunk:
    return DocumentChunk(
        document_id=doc.id,
        chunk_index=index,
        text=f"{doc.title} chunk {index}",
        embedding=_VECTOR,
        token_count=5,
        metadata_=metadata or {},
    )


@pytest.fixture
async def other_document(db_session: AsyncSession, document: Document) -> Document:
    doc = Document(
        workspace_id=document.workspace_id,
        title="Other",
        content_type=ContentType.PLAINTEXT,
        file_path="/tmp/other.txt",
        file_size_bytes=100,
        uploaded_by=document.uploaded_by,
        status=DocumentStatus.INDEXED,
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


@pytest.fixture
async def stored(
    db_session: AsyncSession, document: Document, other_document: Document
) -> None:
    # Chunks 0-4, inserted out of order; the other document's positions overlap, so
    # only the document filter separates them.
    db_session.add_all(
        [_chunk(document, i) for i in (3, 0, 4, 1)]
        + [_chunk(document, 2, metadata={"section": "## Setup"})]
        + [_chunk(other_document, i) for i in range(7)]
    )
    await db_session.flush()


@pytest.mark.usefixtures("stored")
async def test_read_page_returns_the_documents_chunks_in_order_from_start(
    db_session: AsyncSession, document: Document
) -> None:
    repo = SQLAlchemyChunkRepository(db_session)

    page = await repo.read_page(document.id, start=1, limit=3)

    assert page == [
        ChunkText(chunk_index=1, section=None, chunk_text="Runbook chunk 1"),
        ChunkText(chunk_index=2, section="## Setup", chunk_text="Runbook chunk 2"),
        ChunkText(chunk_index=3, section=None, chunk_text="Runbook chunk 3"),
    ]


@pytest.mark.usefixtures("stored")
async def test_read_page_past_the_end_is_empty(
    db_session: AsyncSession, document: Document
) -> None:
    repo = SQLAlchemyChunkRepository(db_session)

    assert await repo.read_page(document.id, start=5, limit=3) == []


@pytest.mark.usefixtures("stored")
async def test_count_counts_the_documents_chunks(
    db_session: AsyncSession, document: Document, other_document: Document
) -> None:
    repo = SQLAlchemyChunkRepository(db_session)

    assert await repo.count(document.id) == 5
    assert await repo.count(other_document.id) == 7


async def test_empty_section_reads_as_none(
    db_session: AsyncSession, document: Document
) -> None:
    # Plaintext and PDF chunks carry no section; markdown's preamble carries "".
    db_session.add(_chunk(document, 0, metadata={"section": ""}))
    await db_session.flush()

    page = await SQLAlchemyChunkRepository(db_session).read_page(
        document.id, start=0, limit=1
    )

    assert [c.section for c in page] == [None]
