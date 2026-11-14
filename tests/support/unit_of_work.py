"""A unit of work over the test's own repository doubles, for service unit tests.

Services take a unit-of-work factory (ADR-033); this one is the factory and the unit
of work at once. It records what a service did with it, not what a database would:
commit and rollback semantics are proved on PostgreSQL (tests/integration).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class FakeUnitOfWork:
    def __init__(
        self,
        *,
        users: Any = None,
        workspaces: Any = None,
        documents: Any = None,
        chunks: Any = None,
        search: Any = None,
        api_keys: Any = None,
    ) -> None:
        self.users = users
        self.workspaces = workspaces
        self.documents = documents
        self.chunks = chunks
        self.search = search
        self.api_keys = api_keys
        self.opened = 0
        self.commits = 0
        # Commits made in a block that then raised: a real unit of work keeps them,
        # but a service committing before its last check is usually a bug.
        self.open_now = False

    async def commit(self) -> None:
        self.commits += 1

    @asynccontextmanager
    async def _block(self) -> AsyncIterator[FakeUnitOfWork]:
        assert not self.open_now, "a unit of work opened inside another (ADR-033)"
        self.opened += 1
        self.open_now = True
        try:
            yield self
        finally:
            self.open_now = False

    def __call__(self) -> Any:
        return self._block()
