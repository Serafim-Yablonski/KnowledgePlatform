"""Full-stack API tests for document endpoints."""

import io

from httpx import AsyncClient

_REGISTER = "/api/v1/auth/register"
_LOGIN = "/api/v1/auth/login"
_WORKSPACES = "/api/v1/workspaces"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _register_and_token(
    client: AsyncClient, email: str, password: str = "password123"
) -> str:
    await client.post(_REGISTER, json={"email": email, "password": password})
    resp = await client.post(_LOGIN, json={"email": email, "password": password})
    return resp.json()["access_token"]  # type: ignore[no-any-return]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _create_workspace(
    client: AsyncClient, token: str, name: str = "Docs WS"
) -> str:
    resp = await client.post(_WORKSPACES, json={"name": name}, headers=_auth(token))
    assert resp.status_code == 201
    return resp.json()["id"]  # type: ignore[no-any-return]


def _docs_url(workspace_id: str) -> str:
    return f"{_WORKSPACES}/{workspace_id}/documents"


def _pdf_file(content: bytes = b"%PDF-1.4 fake") -> dict[str, object]:
    return {"file": ("test.pdf", io.BytesIO(content), "application/pdf")}


# ---------------------------------------------------------------------------
# Upload — happy path
# ---------------------------------------------------------------------------


async def test_upload_pdf_returns_201_with_pending_status(
    async_client: AsyncClient, tmp_path: object
) -> None:
    token = await _register_and_token(async_client, "uploader@example.com")
    ws_id = await _create_workspace(async_client, token)

    resp = await async_client.post(
        _docs_url(ws_id),
        headers=_auth(token),
        data={"title": "My PDF"},
        files=_pdf_file(),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "My PDF"
    assert body["status"] == "pending"
    assert body["content_type"] == "pdf"
    assert body["version"] == 1
    assert "id" in body


# ---------------------------------------------------------------------------
# Upload — validation failures
# ---------------------------------------------------------------------------


async def test_upload_over_size_limit_returns_422(async_client: AsyncClient) -> None:
    token = await _register_and_token(async_client, "big@example.com")
    ws_id = await _create_workspace(async_client, token, "Big File WS")

    from src.domain.documents import MAX_UPLOAD_SIZE_BYTES

    oversized = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)
    resp = await async_client.post(
        _docs_url(ws_id),
        headers=_auth(token),
        data={"title": "Huge"},
        files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
    )
    assert resp.status_code == 422


async def test_upload_unsupported_mime_returns_422(async_client: AsyncClient) -> None:
    token = await _register_and_token(async_client, "mime@example.com")
    ws_id = await _create_workspace(async_client, token, "MIME WS")

    resp = await async_client.post(
        _docs_url(ws_id),
        headers=_auth(token),
        data={"title": "Image"},
        files={"file": ("photo.png", io.BytesIO(b"\x89PNG"), "image/png")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Auth guards
# ---------------------------------------------------------------------------


async def test_upload_requires_auth(async_client: AsyncClient) -> None:
    fake_ws = "00000000-0000-0000-0000-000000000001"
    resp = await async_client.post(
        _docs_url(fake_ws),
        data={"title": "Doc"},
        files=_pdf_file(),
    )
    assert resp.status_code == 403


async def test_non_member_cannot_upload(async_client: AsyncClient) -> None:
    token_owner = await _register_and_token(async_client, "owner_doc@example.com")
    token_stranger = await _register_and_token(async_client, "stranger_doc@example.com")
    ws_id = await _create_workspace(async_client, token_owner, "Owner WS")

    resp = await async_client.post(
        _docs_url(ws_id),
        headers=_auth(token_stranger),
        data={"title": "Steal"},
        files=_pdf_file(),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Full CRUD cycle
# ---------------------------------------------------------------------------


async def test_full_crud_cycle(async_client: AsyncClient) -> None:
    token = await _register_and_token(async_client, "crud@example.com")
    ws_id = await _create_workspace(async_client, token, "CRUD WS")
    url = _docs_url(ws_id)

    # Create
    create_resp = await async_client.post(
        url,
        headers=_auth(token),
        data={"title": "Original Title"},
        files=_pdf_file(),
    )
    assert create_resp.status_code == 201
    doc_id = create_resp.json()["id"]

    # List — document appears
    list_resp = await async_client.get(url, headers=_auth(token))
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(d["id"] == doc_id for d in items)

    # Get
    get_resp = await async_client.get(f"{url}/{doc_id}", headers=_auth(token))
    assert get_resp.status_code == 200
    assert get_resp.json()["title"] == "Original Title"

    # Update (PATCH)
    patch_resp = await async_client.patch(
        f"{url}/{doc_id}",
        headers=_auth(token),
        json={"title": "Updated Title"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Updated Title"
    assert patch_resp.json()["version"] == 2

    # Delete
    del_resp = await async_client.delete(f"{url}/{doc_id}", headers=_auth(token))
    assert del_resp.status_code == 204

    # Get after delete returns 404
    gone_resp = await async_client.get(f"{url}/{doc_id}", headers=_auth(token))
    assert gone_resp.status_code == 404


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


async def test_pagination_follow_cursors_retrieves_all(
    async_client: AsyncClient,
) -> None:
    token = await _register_and_token(async_client, "pages@example.com")
    ws_id = await _create_workspace(async_client, token, "Paged WS")
    url = _docs_url(ws_id)

    # Create 5 documents
    for i in range(5):
        resp = await async_client.post(
            url,
            headers=_auth(token),
            data={"title": f"Doc {i}"},
            files=_pdf_file(),
        )
        assert resp.status_code == 201

    # Paginate with limit=2, follow cursors
    all_ids: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, object] = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page_resp = await async_client.get(url, headers=_auth(token), params=params)
        assert page_resp.status_code == 200
        body = page_resp.json()
        all_ids.extend(d["id"] for d in body["items"])
        cursor = body["next_cursor"]
        if not body["has_more"]:
            break

    assert len(all_ids) == 5
    assert len(set(all_ids)) == 5  # no duplicates


async def test_list_returns_has_more_false_when_no_more_pages(
    async_client: AsyncClient,
) -> None:
    token = await _register_and_token(async_client, "hasmore@example.com")
    ws_id = await _create_workspace(async_client, token, "HasMore WS")
    url = _docs_url(ws_id)

    # Create 3 docs, list with limit=10
    for i in range(3):
        await async_client.post(
            url,
            headers=_auth(token),
            data={"title": f"Doc {i}"},
            files=_pdf_file(),
        )

    resp = await async_client.get(url, headers=_auth(token), params={"limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None


# ---------------------------------------------------------------------------
# Status filter
# ---------------------------------------------------------------------------


async def test_status_filter_query_param(async_client: AsyncClient) -> None:
    token = await _register_and_token(async_client, "statusfilter@example.com")
    ws_id = await _create_workspace(async_client, token, "Status WS")
    url = _docs_url(ws_id)

    await async_client.post(
        url,
        headers=_auth(token),
        data={"title": "A Doc"},
        files=_pdf_file(),
    )

    # Filter by pending — the newly uploaded doc should appear
    resp = await async_client.get(
        url, headers=_auth(token), params={"status": "pending"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["items"]) >= 1

    # Filter by ready — nothing yet
    resp_ready = await async_client.get(
        url, headers=_auth(token), params={"status": "ready"}
    )
    assert resp_ready.status_code == 200
    assert resp_ready.json()["items"] == []


# ---------------------------------------------------------------------------
# Role-based authorization
# ---------------------------------------------------------------------------


async def test_viewer_cannot_upload_update_or_delete(async_client: AsyncClient) -> None:
    """VIEWER role lacks create/update/delete document permissions."""
    token_owner = await _register_and_token(async_client, "owner_rbac@example.com")
    token_viewer = await _register_and_token(async_client, "viewer_rbac@example.com")
    ws_id = await _create_workspace(async_client, token_owner, "RBAC WS")
    url = _docs_url(ws_id)

    # Owner uploads a document
    create_resp = await async_client.post(
        url,
        headers=_auth(token_owner),
        data={"title": "Owner Doc"},
        files=_pdf_file(),
    )
    assert create_resp.status_code == 201
    doc_id = create_resp.json()["id"]

    # Owner adds a VIEWER
    await async_client.post(
        f"{_WORKSPACES}/{ws_id}/members",
        headers=_auth(token_owner),
        json={"user_email": "viewer_rbac@example.com", "role": "viewer"},
    )

    # VIEWER cannot upload
    upload_resp = await async_client.post(
        url,
        headers=_auth(token_viewer),
        data={"title": "Viewer Upload"},
        files=_pdf_file(),
    )
    assert upload_resp.status_code == 403

    # VIEWER cannot update
    patch_resp = await async_client.patch(
        f"{url}/{doc_id}",
        headers=_auth(token_viewer),
        json={"title": "Viewer Rename"},
    )
    assert patch_resp.status_code == 403

    # VIEWER cannot delete
    del_resp = await async_client.delete(f"{url}/{doc_id}", headers=_auth(token_viewer))
    assert del_resp.status_code == 403

    # VIEWER can still read
    get_resp = await async_client.get(f"{url}/{doc_id}", headers=_auth(token_viewer))
    assert get_resp.status_code == 200


# ---------------------------------------------------------------------------
# Cursor validation
# ---------------------------------------------------------------------------


async def test_invalid_cursor_returns_422(async_client: AsyncClient) -> None:
    token = await _register_and_token(async_client, "cursor422@example.com")
    ws_id = await _create_workspace(async_client, token, "Cursor WS")

    resp = await async_client.get(
        _docs_url(ws_id),
        headers=_auth(token),
        params={"cursor": "not-valid-base64!!"},
    )
    assert resp.status_code == 422
