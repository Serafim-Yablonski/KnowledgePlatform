"""API test fixtures — httpx.AsyncClient against the full async FastAPI stack."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import factory
import pytest
from httpx import AsyncClient

import src.services.document as _doc_svc_module


@pytest.fixture(autouse=True)
def _patch_upload_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect file uploads to a per-test temp directory."""
    monkeypatch.setattr(_doc_svc_module.settings, "UPLOAD_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _stub_celery_dispatch() -> Generator[MagicMock]:
    """Prevent Celery task dispatch from connecting to Redis in API tests."""
    from src.workers.tasks.extract_text import extract_text

    with patch.object(extract_text, "delay") as mock:
        yield mock


class UserFactory(factory.Factory):  # type: ignore[misc]
    class Meta:
        model = dict

    email: str = factory.Sequence(lambda n: f"user{n}@example.com")
    password: str = "password123"
    display_name: str = factory.Sequence(lambda n: f"User {n}")


@pytest.fixture
async def auth_client(async_client: AsyncClient) -> AsyncClient:
    data = UserFactory()
    await async_client.post("/api/v1/auth/register", json=data)
    resp = await async_client.post(
        "/api/v1/auth/login",
        json={"email": data["email"], "password": data["password"]},
    )
    token = resp.json()["access_token"]
    async_client.headers["Authorization"] = f"Bearer {token}"
    return async_client
