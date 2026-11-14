"""The unit of work over SQLAlchemy (ADR-033): one session per service operation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import cached_property

import anyio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.repositories.api_key import SQLAlchemyApiKeyRepository
from src.repositories.chunk import SQLAlchemyChunkRepository
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.protocols import UnitOfWorkFactory, UnitOfWorkProtocol
from src.repositories.search import SQLAlchemySearchRepository
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository


class _Opened:
    """Marks a unit of work in the context of the task that opened it, until it closes.

    A task spawned inside one copies the context, so it sees the marker too: while
    the parent is open, a unit of work there would hold a second connection beside
    the parent's. Once the parent closes, the copy reads closed and lets it open.
    """

    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False


_open: ContextVar[_Opened | None] = ContextVar("unit_of_work_open", default=None)


class NestedUnitOfWorkError(RuntimeError):
    """A unit of work was opened while another was open in the same task, or in the
    task that spawned this one (ADR-033).

    It would hold one connection while waiting for another: under load every request
    does it at once and the pool starves.
    """


class SQLAlchemyUnitOfWork:
    """Every repository on one session; each is built on first use."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @cached_property
    def users(self) -> SQLAlchemyUserRepository:
        return SQLAlchemyUserRepository(self._session)

    @cached_property
    def workspaces(self) -> SQLAlchemyWorkspaceRepository:
        return SQLAlchemyWorkspaceRepository(self._session)

    @cached_property
    def documents(self) -> SQLAlchemyDocumentRepository:
        return SQLAlchemyDocumentRepository(self._session)

    @cached_property
    def chunks(self) -> SQLAlchemyChunkRepository:
        return SQLAlchemyChunkRepository(self._session)

    @cached_property
    def search(self) -> SQLAlchemySearchRepository:
        return SQLAlchemySearchRepository(self._session)

    @cached_property
    def api_keys(self) -> SQLAlchemyApiKeyRepository:
        return SQLAlchemyApiKeyRepository(self._session)

    async def commit(self) -> None:
        await self._session.commit()


def unit_of_work_factory(engine: AsyncEngine) -> UnitOfWorkFactory:
    """Units of work on ``engine``. Results stay readable after the block: commit
    doesn't expire them, and closing the session detaches them."""
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def open_unit_of_work() -> AsyncIterator[UnitOfWorkProtocol]:
        enclosing = _open.get()
        if enclosing is not None and not enclosing.closed:
            raise NestedUnitOfWorkError(
                "a unit of work is already open in this task; pass it to helpers "
                "instead of opening another"
            )
        opened = _Opened()
        token = _open.set(opened)
        session = sessions()
        try:
            yield SQLAlchemyUnitOfWork(session)
        finally:
            # Closing rolls back whatever wasn't committed and returns the connection.
            # Shielded: in a cancelled anyio scope (Starlette's and the MCP SDK's task
            # groups) every await raises again, and an interrupted close would leave
            # the connection checked out.
            try:
                with anyio.CancelScope(shield=True):
                    await session.close()
            finally:
                opened.closed = True
                _open.reset(token)

    return open_unit_of_work
