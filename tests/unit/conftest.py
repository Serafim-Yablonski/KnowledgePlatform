"""Unit test fixtures — Protocol stubs and mocked repository implementations."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

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


@pytest.fixture(autouse=True)
def stub_celery_dispatch() -> Generator[MagicMock]:
    """Prevent Celery task dispatch from connecting to Redis in unit tests."""
    from src.workers.tasks.extract_text import extract_text

    with patch.object(extract_text, "delay") as mock:
        yield mock
