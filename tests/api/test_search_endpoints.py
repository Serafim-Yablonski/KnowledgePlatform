"""Full-stack API tests for workspace search: real SQL, only Gemini's HTTP faked."""

from __future__ import annotations

import json
import uuid

import httpx
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.dependencies import get_http_client
from src.domain.documents import ContentType, DocumentStatus
from src.main import app
from src.models.chunk import EMBEDDING_DIMS, DocumentChunk
from src.models.document import Document
from tests.api.conftest import UserFactory

_WORKSPACES = "/api/v1/workspaces"
_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)


def _gemini(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [{"values": _VECTOR}]})


async def _member_with_workspace(client: AsyncClient) -> tuple[dict[str, str], str]:
    data = UserFactory()
    creds = {"email": data["email"], "password": data["password"]}
    await client.post("/api/v1/auth/register", json=creds)
    token = (await client.post("/api/v1/auth/login", json=creds)).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.post(_WORKSPACES, json={"name": "Search WS"}, headers=headers)
    return headers, resp.json()["id"]


class TestSearchEndpoint:
    async def test_hit_locates_the_chunk_in_its_document(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        headers, ws_id = await _member_with_workspace(async_client)
        me = (await async_client.get("/api/v1/auth/me", headers=headers)).json()
        doc = Document(
            workspace_id=uuid.UUID(ws_id),
            title="Runbook",
            content_type=ContentType.MARKDOWN,
            file_path=f"/tmp/{uuid.uuid4()}.md",
            file_size_bytes=100,
            uploaded_by=uuid.UUID(me["id"]),
            status=DocumentStatus.INDEXED,
            version=1,
        )
        db_session.add(doc)
        await db_session.flush()
        db_session.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=3,
                text="Rotate the credentials first.",
                embedding=_VECTOR,
                token_count=5,
                metadata_={"section": "Incidents > Credentials"},
                version=1,
            )
        )
        await db_session.flush()
        gemini = httpx.AsyncClient(transport=httpx.MockTransport(_gemini))
        app.dependency_overrides[get_http_client] = lambda: gemini

        resp = await async_client.post(
            f"{_WORKSPACES}/{ws_id}/search",
            json={"query": "credential rotation"},
            headers=headers,
        )
        await gemini.aclose()

        assert resp.status_code == 200
        assert resp.json()["results"] == [
            {
                "document_id": str(doc.id),
                "document_title": "Runbook",
                "chunk_index": 3,
                "section": "Incidents > Credentials",
                "chunk_text": "Rotate the credentials first.",
                "score": 1.0,
            }
        ]

    async def test_query_is_capped_and_stripped(
        self, async_client: AsyncClient
    ) -> None:
        # ADR-022: 2,000 characters after stripping; what is embedded is stripped.
        headers, ws_id = await _member_with_workspace(async_client)
        embedded: list[str] = []

        def gemini(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            embedded.extend(r["content"]["parts"][0]["text"] for r in body["requests"])
            return _gemini(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(gemini))
        app.dependency_overrides[get_http_client] = lambda: client
        url = f"{_WORKSPACES}/{ws_id}/search"

        statuses = [
            (
                await async_client.post(url, json={"query": q}, headers=headers)
            ).status_code
            for q in ["  " + "x" * 2000 + "\n", "x" * 2001, " \t\n"]
        ]
        await client.aclose()

        assert statuses == [200, 422, 422]
        assert embedded == ["x" * 2000]
