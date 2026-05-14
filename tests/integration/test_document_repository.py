"""Integration tests for SQLAlchemyDocumentRepository against real PostgreSQL 18."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import ContentType, Cursor, DocumentStatus
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.auth import UserCreate
from src.schemas.document import DocumentUpdate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    return await repo.create(
        UserCreate(email=f"doc-int-{uuid.uuid4()}@example.com", password="password123"),
        hashed_password="$2b$12$testhash",
    )


@pytest.fixture
async def test_workspace(db_session: AsyncSession, test_user: User) -> Workspace:
    repo = SQLAlchemyWorkspaceRepository(db_session)
    ws = await repo.create(
        name=f"Test WS {uuid.uuid4().hex[:6]}",
        slug=f"test-ws-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await db_session.flush()
    return ws


async def _create_doc(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    uploaded_by: uuid.UUID,
    *,
    title: str = "Doc",
    status: DocumentStatus = DocumentStatus.PENDING,
    created_at_offset_seconds: float = 0,
) -> Document:
    repo = SQLAlchemyDocumentRepository(session)
    doc = await repo.create(
        workspace_id=workspace_id,
        title=title,
        content_type=ContentType.PLAINTEXT,
        file_path=f"/tmp/{uuid.uuid4()}.txt",
        file_size_bytes=100,
        uploaded_by=uploaded_by,
    )
    if status != DocumentStatus.PENDING:
        doc.status = status
    if created_at_offset_seconds:
        doc.created_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(
            seconds=created_at_offset_seconds
        )
    # flush() is sufficient in the savepoint-based test session; commit() would
    # cycle the SAVEPOINT which breaks in loops of many iterations.
    await session.flush()
    await session.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Cursor-based pagination
# ---------------------------------------------------------------------------


async def test_cursor_pagination_25_docs(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Page 1 returns 20 docs + next_cursor; page 2 returns 5 docs + no cursor."""
    for i in range(25):
        await _create_doc(
            db_session,
            test_workspace.id,
            test_user.id,
            title=f"Doc {i:02d}",
            created_at_offset_seconds=float(i),
        )

    repo = SQLAlchemyDocumentRepository(db_session)
    page1, cursor1 = await repo.list_by_workspace(test_workspace.id, limit=20)
    assert len(page1) == 20
    assert cursor1 is not None

    page2, cursor2 = await repo.list_by_workspace(
        test_workspace.id, limit=20, cursor=cursor1
    )
    assert len(page2) == 5
    assert cursor2 is None

    # No overlap between pages
    ids1 = {d.id for d in page1}
    ids2 = {d.id for d in page2}
    assert ids1.isdisjoint(ids2)


async def test_cursor_pagination_covers_all_docs(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Following cursors exhaustively retrieves all 25 documents."""
    for i in range(25):
        await _create_doc(
            db_session,
            test_workspace.id,
            test_user.id,
            title=f"Doc {i:02d}",
            created_at_offset_seconds=float(i),
        )

    repo = SQLAlchemyDocumentRepository(db_session)
    all_ids: set[uuid.UUID] = set()
    cursor: Cursor | None = None
    while True:
        page, cursor = await repo.list_by_workspace(
            test_workspace.id, limit=10, cursor=cursor
        )
        all_ids.update(d.id for d in page)
        if cursor is None:
            break

    assert len(all_ids) == 25


async def test_cursor_pagination_stability(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    """Inserting a new doc does not corrupt an existing cursor's next page."""
    # Create 22 docs spread over time
    for i in range(22):
        await _create_doc(
            db_session,
            test_workspace.id,
            test_user.id,
            title=f"Doc {i:02d}",
            created_at_offset_seconds=float(i),
        )

    repo = SQLAlchemyDocumentRepository(db_session)
    page1, cursor1 = await repo.list_by_workspace(test_workspace.id, limit=20)
    assert len(page1) == 20
    assert cursor1 is not None

    # Insert a new doc (it will be at the front, before the cursor)
    await _create_doc(
        db_session,
        test_workspace.id,
        test_user.id,
        title="New Doc",
        created_at_offset_seconds=100.0,  # newest, so before cursor in DESC order
    )

    # Existing cursor must still return exactly 2 (new insert is before the cursor)
    page2, cursor2 = await repo.list_by_workspace(
        test_workspace.id, limit=20, cursor=cursor1
    )
    assert len(page2) == 2
    assert cursor2 is None


# ---------------------------------------------------------------------------
# Status filter
# ---------------------------------------------------------------------------


async def test_status_filter_returns_only_matching(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    await _create_doc(
        db_session, test_workspace.id, test_user.id, status=DocumentStatus.PENDING
    )
    await _create_doc(
        db_session, test_workspace.id, test_user.id, status=DocumentStatus.READY
    )
    await _create_doc(
        db_session, test_workspace.id, test_user.id, status=DocumentStatus.FAILED
    )

    repo = SQLAlchemyDocumentRepository(db_session)

    ready_docs, _ = await repo.list_by_workspace(
        test_workspace.id, status=DocumentStatus.READY
    )
    assert len(ready_docs) == 1
    assert ready_docs[0].status == DocumentStatus.READY

    pending_docs, _ = await repo.list_by_workspace(
        test_workspace.id, status=DocumentStatus.PENDING
    )
    assert len(pending_docs) == 1

    all_docs, _ = await repo.list_by_workspace(test_workspace.id)
    assert len(all_docs) == 3


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def test_create_and_get_by_id(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    repo = SQLAlchemyDocumentRepository(db_session)
    doc = await repo.create(
        workspace_id=test_workspace.id,
        title="Hello",
        content_type=ContentType.PDF,
        file_path="/tmp/hello.pdf",
        file_size_bytes=2048,
        uploaded_by=test_user.id,
    )
    doc_id = doc.id  # save before flush/commit expiry
    await db_session.flush()

    fetched = await repo.get_by_id(doc_id)
    assert fetched is not None
    assert fetched.title == "Hello"
    assert fetched.status == DocumentStatus.PENDING
    assert fetched.version == 1


async def test_update_increments_version(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    doc = await _create_doc(db_session, test_workspace.id, test_user.id)
    repo = SQLAlchemyDocumentRepository(db_session)
    updated = await repo.update(doc, DocumentUpdate(title="Updated"))
    assert updated.title == "Updated"
    assert updated.version == 2


async def test_delete_removes_record(
    db_session: AsyncSession, test_workspace: Workspace, test_user: User
) -> None:
    doc = await _create_doc(db_session, test_workspace.id, test_user.id)
    doc_id = doc.id
    repo = SQLAlchemyDocumentRepository(db_session)
    await repo.delete(doc)
    assert await repo.get_by_id(doc_id) is None
