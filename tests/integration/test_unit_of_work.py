"""The unit of work on real PostgreSQL (ADR-033, requirement R1).

Real commits on the shared test database (``committing_engine``); a pool-event
counter tracks the connections held at each point.
"""

from __future__ import annotations

import asyncio
import uuid

import anyio
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.documents import ContentType
from src.domain.user import UserCreationInput
from src.models.document import Document
from src.models.workspace import Workspace
from src.repositories.protocols import UnitOfWorkFactory, UnitOfWorkProtocol
from src.repositories.unit_of_work import NestedUnitOfWorkError, unit_of_work_factory
from tests.support.connections import HeldConnections


@pytest.fixture
def uow(committing_engine: AsyncEngine) -> UnitOfWorkFactory:
    return unit_of_work_factory(committing_engine)


async def _user(uow: UnitOfWorkFactory) -> uuid.UUID:
    async with uow() as u:
        user = await u.users.create(
            UserCreationInput(email=f"{uuid.uuid4().hex}@example.com"), "!"
        )
        await u.commit()
    return user.id


async def _workspaces(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        return list((await conn.scalars(sa.select(Workspace.name))).all())


async def _documents(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return int(await conn.scalar(sa.select(sa.func.count()).select_from(Document)))


async def _create_workspace(
    u: UnitOfWorkProtocol, user_id: uuid.UUID, name: str = "ws"
) -> Workspace:
    return await u.workspaces.create(
        name=name, slug=f"{name}-{uuid.uuid4().hex[:8]}", created_by_id=user_id
    )


# ---------------------------------------------------------------------------
# Commit and rollback
# ---------------------------------------------------------------------------


async def test_commit_persists_and_results_stay_readable(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    user_id = await _user(uow)

    async with uow() as u:
        ws = await _create_workspace(u, user_id, "kept")
        await u.commit()

    assert await _workspaces(committing_engine) == ["kept"]
    # Detached once the block ends, but loaded: no session needed to read it.
    assert ws.name == "kept"
    assert held_connections.now == 0


async def test_leaving_without_commit_rolls_back(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    user_id = await _user(uow)

    async with uow() as u:
        await _create_workspace(u, user_id, "forgotten")

    assert await _workspaces(committing_engine) == []
    assert held_connections.now == 0


async def test_writes_through_different_repositories_commit_together(
    uow: UnitOfWorkFactory, committing_engine: AsyncEngine
) -> None:
    # One session behind every repository: the document's insert needs the
    # workspace's uncommitted row, and neither survives without commit().
    user_id = await _user(uow)

    async with uow() as u:
        ws = await _create_workspace(u, user_id)
        await u.documents.create(
            workspace_id=ws.id,
            title="doc",
            content_type=ContentType.MARKDOWN,
            file_path="ws/doc.md",
            file_size_bytes=1,
            uploaded_by=user_id,
            original_filename=None,
            trace_context=None,
        )

    assert await _workspaces(committing_engine) == []
    assert await _documents(committing_engine) == 0


async def test_an_exception_rolls_back_and_propagates_unchanged(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    user_id = await _user(uow)
    error = ValueError("boom")

    with pytest.raises(ValueError) as raised:
        async with uow() as u:
            await _create_workspace(u, user_id)
            raise error

    assert raised.value is error
    assert await _workspaces(committing_engine) == []
    assert held_connections.now == 0


async def test_a_cancelled_task_rolls_back_and_returns_its_connection(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    user_id = await _user(uow)
    inside = asyncio.Event()

    async def _stuck() -> None:
        async with uow() as u:
            await _create_workspace(u, user_id)
            inside.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(_stuck())
    await inside.wait()
    assert held_connections.now == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert held_connections.now == 0
    assert await _workspaces(committing_engine) == []


async def test_a_cancelled_scope_rolls_back_and_returns_its_connection(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    # anyio's cancellation (Starlette's and the MCP SDK's task groups) is level-
    # triggered: every await inside a cancelled scope raises again, the cleanup's too.
    user_id = await _user(uow)

    with anyio.CancelScope() as scope:
        async with uow() as u:
            await _create_workspace(u, user_id)
            assert held_connections.now == 1
            scope.cancel()
            await anyio.sleep(1)

    assert scope.cancelled_caught
    assert held_connections.now == 0
    assert await _workspaces(committing_engine) == []


# ---------------------------------------------------------------------------
# No nesting (ADR-033, D4)
# ---------------------------------------------------------------------------


async def test_a_unit_of_work_inside_another_raises(
    uow: UnitOfWorkFactory, held_connections: HeldConnections
) -> None:
    async with uow() as u:
        await u.users.get_by_email("nobody@example.com")
        with pytest.raises(NestedUnitOfWorkError):
            async with uow():
                pass
        # The refused one took no connection; the outer one keeps working.
        assert held_connections.peak == 1
        await u.users.get_by_email("nobody@example.com")

    assert held_connections.now == 0


async def test_a_task_spawned_inside_one_cannot_open_another(
    uow: UnitOfWorkFactory,
) -> None:
    async def _open() -> None:
        async with uow():
            pass

    async with uow():
        with pytest.raises(NestedUnitOfWorkError):
            await asyncio.create_task(_open())


async def test_a_task_spawned_inside_one_may_open_its_own_once_that_one_closed(
    uow: UnitOfWorkFactory,
) -> None:
    # Fire-and-forget (ResearchService.start's shape): the task copied the open
    # flag, but by the time it runs its parent holds no connection.
    parent_closed = asyncio.Event()

    async def _later() -> None:
        await parent_closed.wait()
        async with uow() as u:
            await u.users.get_by_email("nobody@example.com")

    async with uow():
        task = asyncio.create_task(_later())
    parent_closed.set()

    await task


async def test_sibling_tasks_each_open_their_own(
    uow: UnitOfWorkFactory,
    committing_engine: AsyncEngine,
    held_connections: HeldConnections,
) -> None:
    # What parallel tool calls do (R3): concurrent, not nested, so both run, each
    # on its own connection.
    user_id = await _user(uow)
    both_inside = asyncio.Barrier(2)

    async def _write(name: str) -> None:
        async with uow() as u:
            await _create_workspace(u, user_id, name)
            await both_inside.wait()
            await u.commit()

    await asyncio.gather(_write("a"), _write("b"))

    assert sorted(await _workspaces(committing_engine)) == ["a", "b"]
    assert held_connections.peak == 2
    assert held_connections.now == 0


async def test_one_after_another_in_the_same_task(uow: UnitOfWorkFactory) -> None:
    # The guard is released when a unit of work closes, however it closed.
    with pytest.raises(ValueError):
        async with uow():
            raise ValueError
    async with uow():
        pass
    async with uow():
        pass
