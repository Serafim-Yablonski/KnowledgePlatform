"""Log output: development tracebacks show no frame's local variables (a password
or key held in one would be printed with it)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import structlog

from src.core.logging import setup_logging


@pytest.fixture
def development_logging() -> Iterator[None]:
    config = structlog.get_config()
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    setup_logging(environment="development")
    try:
        yield
    finally:
        structlog.configure(**config)
        root.handlers, root.level = handlers, level


def _login(password: str) -> None:
    raise ValueError("password backend unavailable")


@pytest.mark.usefixtures("development_logging")
def test_a_development_traceback_shows_no_local_variables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Built at run time: rich also prints the source lines around each frame.
    secret = "-".join(["fake", "user", "password"])
    try:
        _login(secret)
    except ValueError:
        structlog.get_logger("test").exception("login failed")

    out = capsys.readouterr().out
    assert "password backend unavailable" in out  # the traceback is still there
    assert secret not in out
