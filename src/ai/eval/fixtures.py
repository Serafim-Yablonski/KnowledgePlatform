"""Eval workspace setup and teardown utilities.

Bypasses Celery by performing chunking and embedding inline so the eval runner
works without a running worker. Creates a fully isolated workspace that can be
cleaned up after each eval run.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import sqlalchemy as sa
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.chunking.factory import get_chunker
from src.ai.embeddings import EmbeddingService
from src.domain.documents import ContentType, DocumentStatus
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace

logger = structlog.get_logger(__name__)

_EVAL_USER_EMAIL = "eval-system@internal.invalid"
_EVAL_USER_PASSWORD_HASH = (  # not a real bcrypt hash — eval user cannot log in
    "$2b$12$evaluationsystemhashplaceholder0000000000000000000000000"
)

_SLUG_TO_CONTENT_TYPE: dict[str, ContentType] = {
    ".md": ContentType.MARKDOWN,
    ".txt": ContentType.PLAINTEXT,
}


async def _get_or_create_eval_user(session: AsyncSession) -> User:
    result = await session.scalars(
        sa.select(User).where(User.email == _EVAL_USER_EMAIL)
    )
    existing = result.first()
    if existing is not None:
        return existing

    user = User(
        email=_EVAL_USER_EMAIL,
        hashed_password=_EVAL_USER_PASSWORD_HASH,
        display_name="Eval System",
        is_active=True,
    )
    session.add(user)
    await session.flush()
    await session.refresh(user)
    return user


async def load_test_documents(
    workspace_id: uuid.UUID,
    docs_dir: Path,
    session: AsyncSession,
    embedding_service: EmbeddingService,
) -> dict[str, uuid.UUID]:
    """Load test documents into the eval workspace and embed them inline.

    Returns a mapping of doc-slug (filename without extension) → document UUID.
    The caller can use this map to translate golden dataset source IDs to UUIDs.
    """
    eval_user = await _get_or_create_eval_user(session)
    slug_to_id: dict[str, uuid.UUID] = {}

    doc_files = sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        logger.warning("no test documents found", docs_dir=str(docs_dir))
        return slug_to_id

    for doc_file in doc_files:
        slug = doc_file.stem
        content_type = _SLUG_TO_CONTENT_TYPE.get(doc_file.suffix, ContentType.PLAINTEXT)
        raw_text = doc_file.read_text(encoding="utf-8")

        doc = Document(
            workspace_id=workspace_id,
            title=slug.replace("-", " ").title(),
            content_type=content_type,
            file_path=str(doc_file),
            file_size_bytes=len(raw_text.encode()),
            uploaded_by=eval_user.id,
            status=DocumentStatus.READY,
            version=1,
        )
        doc.raw_text = raw_text
        session.add(doc)
        await session.flush()
        await session.refresh(doc)

        chunker = get_chunker(content_type)
        chunk_data_list = chunker.chunk(raw_text, {"document_id": str(doc.id)})

        if not chunk_data_list:
            logger.warning("no chunks produced", slug=slug, doc_id=str(doc.id))
            slug_to_id[slug] = doc.id
            continue

        texts = [cd.text for cd in chunk_data_list]
        embeddings = await embedding_service.embed_texts(texts)

        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                text=cd.text,
                embedding=embedding,
                token_count=cd.token_count,
                metadata_=cd.metadata,
                version=1,
            )
            for i, (cd, embedding) in enumerate(
                zip(chunk_data_list, embeddings, strict=True)
            )
        ]
        session.add_all(chunks)
        await session.flush()

        slug_to_id[slug] = doc.id
        logger.info(
            "loaded eval document",
            slug=slug,
            doc_id=str(doc.id),
            chunks=len(chunks),
        )

    await session.commit()
    logger.info("eval workspace loaded", document_count=len(slug_to_id))
    return slug_to_id


async def setup_eval_workspace(
    session: AsyncSession,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create an isolated workspace for eval runs.

    Returns (workspace_id, eval_user_id). The workspace has no members — the
    eval runner accesses it directly via the search repository, bypassing the
    membership check in WorkspaceService.
    """
    user_id = await _get_eval_placeholder_user_id(session)
    workspace = Workspace(
        name="Eval Workspace",
        slug=f"eval-{uuid.uuid4().hex[:8]}",
        description="Isolated workspace for RAG evaluation. Safe to delete.",
        created_by=user_id,
        is_active=True,
    )
    session.add(workspace)
    await session.flush()
    await session.refresh(workspace)
    await session.commit()
    logger.info("eval workspace created", workspace_id=str(workspace.id))
    return workspace.id, user_id


async def cleanup_stale_eval_workspaces(session: AsyncSession) -> None:
    """Delete any workspaces with slug matching 'eval-%' left by failed prior runs."""
    await session.execute(sa.delete(Workspace).where(Workspace.slug.like("eval-%")))
    await session.commit()
    logger.info("stale eval workspace cleanup ran")


async def _get_eval_placeholder_user_id(session: AsyncSession) -> uuid.UUID:
    """Return eval user ID, creating the user if needed."""
    user = await _get_or_create_eval_user(session)
    return user.id


async def cleanup_eval_workspace(
    workspace_id: uuid.UUID,
    session: AsyncSession,
) -> None:
    """Delete all documents, chunks, and the workspace for the given eval run."""
    # Document and chunk deletions cascade via FK — delete workspace covers all.
    await session.execute(sa.delete(Workspace).where(Workspace.id == workspace_id))
    await session.commit()
    logger.info("eval workspace cleaned up", workspace_id=str(workspace_id))
