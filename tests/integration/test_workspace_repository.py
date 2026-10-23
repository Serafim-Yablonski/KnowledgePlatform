"""Integration tests for SQLAlchemyWorkspaceRepository against real PostgreSQL 18."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.roles import WorkspaceRole
from src.domain.workspace import WorkspaceUpdateInput
from src.models.user import User
from src.models.workspace import WorkspaceMembership
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.auth import UserCreate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    return await repo.create(
        UserCreate(email=f"ws-int-{uuid.uuid4()}@example.com", password="password123"),
        hashed_password="$2b$12$testhash",
    )


@pytest.fixture
async def workspace_repo(db_session: AsyncSession) -> SQLAlchemyWorkspaceRepository:
    return SQLAlchemyWorkspaceRepository(db_session)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


async def test_update_name(
    workspace_repo: SQLAlchemyWorkspaceRepository, test_user: User
) -> None:
    ws = await workspace_repo.create(
        name="Original",
        slug=f"original-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await workspace_repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)

    updated = await workspace_repo.update(ws.id, WorkspaceUpdateInput(name="Renamed"))

    assert updated.id == ws.id
    assert updated.name == "Renamed"
    assert updated.slug == ws.slug  # slug is never touched by update


async def test_update_description(
    workspace_repo: SQLAlchemyWorkspaceRepository, test_user: User
) -> None:
    ws = await workspace_repo.create(
        name="Desc WS",
        slug=f"desc-ws-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
        description=None,
    )
    await workspace_repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)

    updated = await workspace_repo.update(
        ws.id, WorkspaceUpdateInput(description="Added description")
    )

    assert updated.description == "Added description"


async def test_update_no_fields_leaves_workspace_unchanged(
    workspace_repo: SQLAlchemyWorkspaceRepository, test_user: User
) -> None:
    ws = await workspace_repo.create(
        name="Unchanged",
        slug=f"unchanged-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
        description="Keep me",
    )
    await workspace_repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)

    updated = await workspace_repo.update(ws.id, WorkspaceUpdateInput())

    assert updated.name == "Unchanged"
    assert updated.description == "Keep me"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_workspace_removes_row(
    workspace_repo: SQLAlchemyWorkspaceRepository, test_user: User
) -> None:
    ws = await workspace_repo.create(
        name="Delete Me",
        slug=f"delete-me-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await workspace_repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)

    await workspace_repo.delete(ws.id)

    assert await workspace_repo.get_by_id(ws.id) is None


async def test_delete_workspace_cascades_memberships(
    workspace_repo: SQLAlchemyWorkspaceRepository, test_user: User
) -> None:
    ws = await workspace_repo.create(
        name="Cascade Test",
        slug=f"cascade-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await workspace_repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)
    assert await workspace_repo.count_members(ws.id) == 1

    await workspace_repo.delete(ws.id)

    assert await workspace_repo.count_members(ws.id) == 0


async def test_delete_nonexistent_workspace_is_noop(
    workspace_repo: SQLAlchemyWorkspaceRepository,
) -> None:
    await workspace_repo.delete(uuid.uuid4())  # should not raise


async def test_update_persists_to_db(db_session: AsyncSession, test_user: User) -> None:
    repo = SQLAlchemyWorkspaceRepository(db_session)
    ws = await repo.create(
        name="Persist Test",
        slug=f"persist-{uuid.uuid4().hex[:8]}",
        created_by_id=test_user.id,
    )
    await repo.add_member(ws.id, test_user.id, WorkspaceRole.OWNER)

    await repo.update(ws.id, WorkspaceUpdateInput(name="After Update"))

    # Re-fetch from DB to confirm the change was committed
    fetched = await repo.get_by_id(ws.id)
    assert fetched is not None
    assert fetched.name == "After Update"


# ---------------------------------------------------------------------------
# list_for_user_with_counts
# ---------------------------------------------------------------------------


async def _join(
    db_session: AsyncSession,
    repo: SQLAlchemyWorkspaceRepository,
    workspace_id: uuid.UUID,
    user: User,
    day: int,
) -> None:
    """Add ``user`` as joined on a given day: now() is one value per transaction."""
    await repo.add_member(workspace_id, user.id, WorkspaceRole.MEMBER)
    await db_session.execute(
        update(WorkspaceMembership)
        .where(
            WorkspaceMembership.workspace_id == workspace_id,
            WorkspaceMembership.user_id == user.id,
        )
        .values(joined_at=datetime(2026, 1, day))
    )


async def test_list_for_user_with_counts(
    db_session: AsyncSession,
    workspace_repo: SQLAlchemyWorkspaceRepository,
    test_user: User,
) -> None:
    other = await SQLAlchemyUserRepository(db_session).create(
        UserCreate(email=f"ws-other-{uuid.uuid4()}@example.com", password="pw123456"),
        hashed_password="$2b$12$testhash",
    )
    ws = {}
    for name in ("first", "second", "third", "not-mine"):
        ws[name] = await workspace_repo.create(
            name=name, slug=f"{name}-{uuid.uuid4().hex[:8]}", created_by_id=other.id
        )
    await _join(db_session, workspace_repo, ws["first"].id, test_user, day=1)
    await _join(db_session, workspace_repo, ws["third"].id, test_user, day=2)
    await _join(db_session, workspace_repo, ws["second"].id, test_user, day=3)
    # Another member joined most recently: the order follows the caller's joins.
    await _join(db_session, workspace_repo, ws["first"].id, other, day=5)
    await _join(db_session, workspace_repo, ws["not-mine"].id, other, day=4)

    rows = await workspace_repo.list_for_user_with_counts(test_user.id)

    # Only the caller's workspaces, newest join first; the count is every member.
    assert [(w.name, count) for w, count in rows] == [
        ("second", 1),
        ("third", 1),
        ("first", 2),
    ]
