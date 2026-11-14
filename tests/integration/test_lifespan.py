"""The real lifespan (ADR-033): it opens the process's engine and clients, builds one
``Services`` from them, and REST and MCP requests both reach that one through the
lifespan state the server copies into each request.

Starlette's ``TestClient`` serves the app as uvicorn does (lifespan first, its state
copied per request) through the real ``create_app()``, instrumentation included.
Only the steps that reach outside the test are stubbed: the research checkpointer
and process-global logging and telemetry.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.testclient import TestClient

import src.main
from src.core.config import settings
from src.core.security import create_access_token
from src.main import create_app
from tests.support import seed
from tests.support.mcp import rpc_request


@asynccontextmanager
async def _no_checkpointer() -> AsyncGenerator[None]:
    yield None


@pytest.fixture
def built() -> list[Any]:
    """Every ``Services`` the lifespan builds."""
    return []


@pytest.fixture
def lifespan_on_test_db(
    monkeypatch: pytest.MonkeyPatch, test_db_url: str, built: list[Any]
) -> None:
    monkeypatch.setattr(settings, "ASYNC_DATABASE_URL", SecretStr(test_db_url))
    monkeypatch.setattr(src.main, "setup_checkpointer", _no_checkpointer)
    monkeypatch.setattr(src.main, "setup_observability", lambda *a, **kw: None)
    monkeypatch.setattr(src.main, "setup_logging", lambda **kw: None)
    build = src.main.build_services

    def _recorded(*args: Any) -> Any:
        built.append(build(*args))
        return built[-1]

    monkeypatch.setattr(src.main, "build_services", _recorded)


def _call_tool(client: TestClient, token: str, name: str) -> Any:
    """A raw ``tools/call``, shaped as the SDK client sends it."""
    body, headers = rpc_request(
        "tools/call", {"name": name, "arguments": {}}, name=name
    )
    response = client.post(
        "/mcp", json=body, headers={**headers, "Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, response.text
    reply = response.text
    if response.headers["content-type"].startswith("text/event-stream"):
        reply = next(
            line.removeprefix("data:")
            for line in reply.splitlines()
            if line[:5] == "data:"
        )
    return json.loads(reply)


@pytest.mark.usefixtures("lifespan_on_test_db")
async def test_one_services_per_process_serves_rest_and_mcp(
    committing_engine: AsyncEngine, built: list[Any]
) -> None:
    user = await seed.user(committing_engine)
    workspace_id = await seed.workspace_of(committing_engine, user)
    token = create_access_token(user.id)

    with TestClient(create_app()) as client:  # entered: the lifespan runs first
        [services] = built
        # A cached_property lands in the instance's __dict__ on first use: what each
        # request adds there shows it used this very instance.
        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        used_by_rest = set(vars(services))
        listed = _call_tool(client, token, "list_workspaces")
        used_by_mcp = set(vars(services)) - used_by_rest

    assert me.status_code == 200
    assert me.json()["id"] == str(user.id)
    workspaces = listed["result"]["structuredContent"]["workspaces"]
    assert [uuid.UUID(w["id"]) for w in workspaces] == [workspace_id]
    assert "auth" in used_by_rest
    assert "workspace" in used_by_mcp
    assert len(built) == 1
