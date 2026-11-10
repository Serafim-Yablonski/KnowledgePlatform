"""Where transactions begin and end, through the real wiring (ADR-033, R2–R4).

Real commits (``committing_engine``) and a pool-event counter: these tests need two
connections and real transaction boundaries, which the one-connection ``db_session``
harness can't give. They exercise services and routes, not their internals, so the
unit-of-work rewiring (Track 2.5 step 4) changed the fixtures, not these tests.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import AsyncClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.ai.agents.question import agent
from src.composition import Services
from src.core import config
from src.core.exceptions import ConflictError
from src.core.security import create_access_token
from src.domain.documents import DocumentUpload
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.repositories.file_store import LocalFileStore
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from tests.support import seed
from tests.support.connections import EmbeddingProvider, HeldConnections

_ANSWER = {
    "answer": "Nothing found.",
    "sources": [],
    "confidence": 0.0,
    "reasoning": "No documents matched.",
}


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config.settings, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


def _auth(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _upload() -> DocumentUpload:
    return DocumentUpload(
        filename="notes.txt",
        content_type="text/plain",
        declared_size=None,
        stream=io.BytesIO(b"plain text"),
    )


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


# ---------------------------------------------------------------------------
# R2: no connection is held while waiting on anything outside the database
# ---------------------------------------------------------------------------


async def test_search_holds_no_connection_while_the_provider_embeds(
    committing_engine: AsyncEngine,
    committing_client: AsyncClient,
    held_connections: HeldConnections,
    embedding_provider: EmbeddingProvider,
) -> None:
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)
    held_during: list[int] = []
    embedding_provider.on_request = lambda: held_during.append(held_connections.now)

    resp = await committing_client.post(
        f"/api/v1/workspaces/{ws_id}/search", json={"query": "q"}, headers=_auth(user)
    )

    assert resp.status_code == 200
    assert held_during == [0]


async def test_ask_holds_no_connection_while_the_model_works(
    committing_engine: AsyncEngine,
    committing_client: AsyncClient,
    held_connections: HeldConnections,
) -> None:
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)
    doc = await seed.document(committing_engine, ws_id, user)
    held_during: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        held_during.append(held_connections.now)
        if not _tool_returns(messages):  # a tool that reads the database, first
            args = {"document_id": str(doc.id)}
            return ModelResponse(parts=[ToolCallPart("get_document_details", args)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, _ANSWER)])

    with agent.override(model=FunctionModel(model)):
        resp = await committing_client.post(
            f"/api/v1/workspaces/{ws_id}/ai/ask",
            json={"question": "What is in the docs?"},
            headers=_auth(user),
        )

    assert resp.status_code == 200
    assert held_during == [0, 0]


async def test_upload_holds_exactly_one_connection_while_writing_its_file(
    committing_engine: AsyncEngine,
    committing_client: AsyncClient,
    held_connections: HeldConnections,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one exception to R2, on purpose (F5): the upload's row lock is held across
    # its file write so that a workspace delete waits for it.
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)
    held_during: list[int] = []
    real_save = LocalFileStore.save

    async def save(self: LocalFileStore, *args: Any, **kwargs: Any) -> str:
        held_during.append(held_connections.now)
        return await real_save(self, *args, **kwargs)

    monkeypatch.setattr(LocalFileStore, "save", save)

    resp = await committing_client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
        headers=_auth(user),
    )

    assert resp.status_code == 201
    assert held_during == [1]


# ---------------------------------------------------------------------------
# R3: concurrent operations inside one use case are safe
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls_each_read_the_database(
    committing_engine: AsyncEngine, committing_services: Services
) -> None:
    # F3: Anthropic and Gemini models send parallel tool calls by default, and
    # PydanticAI runs them as concurrent tasks.
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)
    docs = [await seed.document(committing_engine, ws_id, user) for _ in range(2)]
    returned: list[ToolReturnPart] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not (returns := _tool_returns(messages)):
            return ModelResponse(
                parts=[
                    ToolCallPart("get_document_details", {"document_id": str(d.id)})
                    for d in docs
                ]
            )
        returned.extend(returns)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, _ANSWER)])

    with agent.override(model=FunctionModel(model)):
        await committing_services.ai.ask(ws_id, user.id, "q", WorkspaceRole.OWNER)

    assert sorted(r.model_response_object()["title"] for r in returned) == sorted(
        d.title for d in docs
    )


# ---------------------------------------------------------------------------
# R4: the orderings any design keeps (F5)
# ---------------------------------------------------------------------------


async def test_a_failed_delete_commit_keeps_the_row_and_its_file(
    committing_engine: AsyncEngine,
    committing_services: Services,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Files go only after the delete committed: a row without its file is corruption,
    # a file without its row only garbage.
    user = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, user)
    doc = await committing_services.document.create(
        user, ws_id, WorkspaceRole.OWNER, None, _upload()
    )
    [stored] = _files(upload_dir)

    async def connection_lost(self: AsyncSession) -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(AsyncSession, "commit", connection_lost)
    with pytest.raises(OperationalError):
        await committing_services.document.delete(
            user, ws_id, WorkspaceRole.OWNER, doc.id
        )
    monkeypatch.undo()

    assert _files(upload_dir) == [stored]
    async with committing_engine.connect() as conn:
        rows = await conn.scalar(
            sa.select(sa.func.count())
            .select_from(Document)
            .where(Document.id == doc.id)
        )
    assert rows == 1


async def test_a_duplicate_member_the_pre_check_missed_is_a_conflict(
    committing_engine: AsyncEngine,
    committing_services: Services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A concurrent add can pass the pre-check; the unique constraint is the real
    # guard. Its IntegrityError must surface inside the service, where it maps to a
    # 409 — a repository that stopped flushing would move it to the commit, a 500.
    owner = await seed.user(committing_engine)
    member = await seed.user(committing_engine)
    ws_id = await seed.workspace_of(committing_engine, owner, member)

    async def missed(self: SQLAlchemyWorkspaceRepository, *_: Any) -> None:
        return None

    monkeypatch.setattr(SQLAlchemyWorkspaceRepository, "get_membership", missed)

    with pytest.raises(ConflictError, match="Invitation could not be sent"):
        await committing_services.workspace.add_member(
            owner, ws_id, WorkspaceRole.OWNER, member.email
        )
