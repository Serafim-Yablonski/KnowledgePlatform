"""Smoke tests — imports every skeleton module and verifies the basic HTTP contract."""

from fastapi.testclient import TestClient

import src.ai.eval.compare
import src.ai.eval.runner
import src.workers.celery_app
from src.core.config import settings
from src.main import app


def test_health_endpoint() -> None:
    # Lifespan doesn't run outside the context manager, so DB/Redis aren't available.
    # The endpoint must still respond with valid JSON and the standard shape.
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_celery_app_exists() -> None:
    assert src.workers.celery_app.celery_app is not None


def test_eval_modules_importable() -> None:
    assert callable(src.ai.eval.runner.main)
    assert callable(src.ai.eval.compare.main)
    src.ai.eval.compare.main("baseline.json", "current.json")


def test_settings_defaults() -> None:
    assert settings.EMBEDDING_DIMENSIONS == 768
    assert settings.LLM_MODEL.startswith("google-gla:")
