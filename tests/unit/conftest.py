"""Unit test fixtures — Protocol stubs and mocked repository implementations."""

import pytest


class _AccidentalDbAccess:
    """Raises immediately if any unit test accidentally reaches the database."""

    def __call__(self, *args: object, **kwargs: object) -> _AccidentalDbAccess:
        raise RuntimeError(
            "Unit tests must not access the database — use Protocol stubs instead."
        )

    async def __aenter__(self) -> _AccidentalDbAccess:
        raise RuntimeError(
            "Unit tests must not access the database — use Protocol stubs instead."
        )

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.fixture
def mock_session_factory() -> _AccidentalDbAccess:
    return _AccidentalDbAccess()
