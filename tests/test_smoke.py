"""Smoke tests — importing the app, and the settings' defaults."""

from src.core.config import settings
from src.main import app  # noqa: F401 — importing builds the app


def test_settings_defaults() -> None:
    assert settings.EMBEDDING_DIMENSIONS == 768
    assert settings.LLM_MODEL.startswith("google-gla:")
