"""Integration tests for SQLAlchemyWorkspaceRepository against real PostgreSQL 18."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.roles import WorkspaceRole
from src.domain.workspace import WorkspaceUpdateInput
from src.models.user import User
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
