"""Protocol-level MCP tests: the real SDK client over ASGI against the real app.

Each test builds a fresh app (a session manager runs once per instance) and starts
its manager with ``running()``. Auth and tools run on real services over the test
DB; Redis is faked and outbound HTTP is refused.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any, NamedTuple
from unittest.mock import ANY

import anyio
import httpx
import httpx2
import jwt
import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI
from httpx import ASGITransport
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult
from mcp.types.jsonrpc import HEADER_MISMATCH, UNSUPPORTED_PROTOCOL_VERSION
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import ASGIApp, Receive, Scope, Send

import src.composition
import src.core.http
import src.core.redis
import src.mcp_server.tools
from src.core.config import settings
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from src.domain.roles import WorkspaceRole
from src.main import create_app
from src.mcp_server.server import build_mcp_server
from src.models.user import User
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.schemas.auth import UserCreate

# Default base URL: a local client against a dev server.
LOCAL_URL = "http://localhost:8000"
# A deployment's real hostname, for tests where a non-localhost Host is the point:
# the SDK's localhost-only Host allowlist (now disabled) rejected it with 421 (F2).
PUBLIC_URL = "http://kp.example.com"

# The only protocol revision we support: every request is self-contained (no
# initialize handshake, no session id) and names its version in a header and _meta.
PROTOCOL_VERSION = "2026-07-28"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class Member(NamedTuple):
    token: str
    workspace_id: uuid.UUID


async def _create_member(db_session: AsyncSession, name: str) -> Member:
    """A user who owns one workspace, with a JWT for it."""
    user = await SQLAlchemyUserRepository(db_session).create(
        UserCreate(email=f"{name}@example.com", password="password123"),
        hashed_password="$2b$12$fixturehash",
    )
    repo = SQLAlchemyWorkspaceRepository(db_session)
    ws = await repo.create(
        name=f"{name}-ws",
        slug=f"{name}-{uuid.uuid4().hex[:8]}",
        created_by_id=user.id,
    )
    await repo.add_member(workspace_id=ws.id, user_id=user.id, role=WorkspaceRole.OWNER)
    return Member(create_access_token(user.id), ws.id)


@pytest.fixture
async def mcp_backends(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[None]:
    """Route the real open_services() — auth and tools alike — to test resources.

    Every unit of work shares the test's one session, so they run one at a time:
    concurrent tool calls would otherwise use the session concurrently, which
    SQLAlchemy rejects.
    """
    one_at_a_time = anyio.Lock()

    @asynccontextmanager
    async def _test_session() -> AsyncGenerator[AsyncSession]:
        async with one_at_a_time:
            yield db_session

    def _no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected outbound HTTP call: {request.url}")

    fake_redis = FakeRedis(decode_responses=True)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_no_network))
    monkeypatch.setattr(src.composition, "get_session", _test_session)
    monkeypatch.setattr(src.core.redis, "_async_client", fake_redis)
    monkeypatch.setattr(src.core.http, "_http_client", http_client)
    yield
    await fake_redis.aclose()
    await http_client.aclose()


@pytest.fixture
def app(mcp_backends: None) -> FastAPI:
    return create_app()


@pytest.fixture
async def alice(db_session: AsyncSession) -> Member:
    return await _create_member(db_session, "alice")


@pytest.fixture
async def bob(db_session: AsyncSession) -> Member:
    return await _create_member(db_session, "bob")


# ---------------------------------------------------------------------------
# Helpers — context managers, not fixtures: an async fixture holding the
# manager's run() across yield fails at teardown ("exit cancel scope in a
# different task").
# ---------------------------------------------------------------------------


class _SwappableBearer(httpx2.Auth):
    """Bearer auth read per request, so a test can switch user between calls."""

    def __init__(self, token: str) -> None:
        self.token = token

    def auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


def _single_leaf(group: BaseExceptionGroup[BaseException]) -> BaseException | None:
    """The only exception in a (nested) group, or None if it holds several."""
    leaves: list[BaseException] = []
    pending: list[BaseException] = [group]
    while pending:
        exc = pending.pop()
        if isinstance(exc, BaseExceptionGroup):
            pending.extend(exc.exceptions)
        else:
            leaves.append(exc)
    return leaves[0] if len(leaves) == 1 else None


@asynccontextmanager
async def running(app: FastAPI) -> AsyncGenerator[None]:
    """Run the app's MCP session manager for one test body, failing it after 10 s.

    The manager is an anyio task group, so anything raised inside it — even a plain
    assert — leaves wrapped in an ExceptionGroup. A lone exception is re-raised
    unwrapped so ``pytest.raises`` and pytest's report see the real error.
    """
    try:
        # Backstop only: no hang has been observed; a transport error fails at once.
        with anyio.fail_after(10):
            async with app.state.mcp_session_manager.run():
                yield
    except BaseExceptionGroup as group:
        leaf = _single_leaf(group)
        if leaf is None:
            raise
        raise leaf from group


@asynccontextmanager
async def mcp_client(
    app: ASGIApp, auth: _SwappableBearer, base_url: str = LOCAL_URL
) -> AsyncGenerator[Client]:
    """An SDK client pinned to PROTOCOL_VERSION; call inside ``running(app)``.

    A version pin connects without any request (no handshake, no discovery probe),
    so every request the client sends is a tool or list call.
    """
    async with (
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            base_url=base_url,
            auth=auth,
            follow_redirects=False,  # a 307 must fail the test, not be followed
        ) as http,
        Client(
            streamable_http_client(f"{base_url}/mcp", http_client=http),
            mode=PROTOCOL_VERSION,
        ) as client,
    ):
        yield client


def _tool_json(result: CallToolResult) -> Any:
    """The tool's structured content (its text block carries the same data)."""
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def _post_rpc(
    app: FastAPI, headers: dict[str, str], base_url: str = LOCAL_URL
) -> httpx.Response:
    """A raw, valid tools/list POST, for tests about HTTP status and headers.

    Shaped as the SDK client sends it: the version and method in headers and the
    version and client details in ``params._meta`` (without them the SDK answers 400).
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
            }
        },
    }
    async with httpx.AsyncClient(
        transport=ASGITransport(app), base_url=base_url, follow_redirects=False
    ) as http:
        return await http.post(
            "/mcp",
            json=body,
            headers={
                # Without this Accept the SDK answers 406 whatever else is wrong.
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
                "MCP-Method": "tools/list",
                **headers,
            },
        )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


async def test_mcp_path_served_without_redirect(app: FastAPI, alice: Member) -> None:
    # Guards the exact Route("/mcp"): a Mount would answer 307 to /mcp/, and httpx
    # drops Authorization when that redirect downgrades https to http.
    async with running(app):
        response = await _post_rpc(app, {"Authorization": f"Bearer {alice.token}"})

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


async def test_server_exposes_only_tools(app: FastAPI, alice: Member) -> None:
    # Workspace selection is a tool argument (no session state), and resources are
    # gone: the model can't call them, and list tools replace them.
    async with running(app), mcp_client(app, _SwappableBearer(alice.token)) as c:
        tools = {tool.name for tool in (await c.list_tools()).tools}
        resources = (await c.list_resources()).resources
        templates = (await c.list_resource_templates()).resource_templates

    assert tools == {
        "list_workspaces",
        "search_documents",
        "list_documents",
        "read_document",
        "start_research",
        "get_research_status",
    }
    assert (resources, templates) == ([], [])


async def test_default_mode_client_discovers_the_server(
    app: FastAPI, alice: Member
) -> None:
    # The SDK client's default mode probes server/discover first; under 2026-07-28
    # that result is the only carrier of the server's instructions. If our middleware
    # refused it, such a client would fall back to the handshake we refuse too.
    async with (
        running(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            base_url=LOCAL_URL,
            auth=_SwappableBearer(alice.token),
            follow_redirects=False,
        ) as http,
        Client(streamable_http_client(f"{LOCAL_URL}/mcp", http_client=http)) as client,
    ):
        assert client.protocol_version == PROTOCOL_VERSION
        assert client.instructions == build_mcp_server().instructions
        assert client.instructions


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def test_units_of_work_never_overlap(
    app: FastAPI, alice: Member, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two concurrent tool calls would use the shared test session at once, which
    # SQLAlchemy rejects; the fixture's lock must serialize them.
    locked_session = src.composition.get_session
    in_flight = peak = 0

    @asynccontextmanager
    async def _counting_session() -> AsyncGenerator[AsyncSession]:
        nonlocal in_flight, peak
        async with locked_session() as session:
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await anyio.sleep(0.05)  # widen the window an overlap would need
                yield session
            finally:
                in_flight -= 1

    monkeypatch.setattr(src.composition, "get_session", _counting_session)

    async with (
        running(app),
        mcp_client(app, _SwappableBearer(alice.token)) as client,
        anyio.create_task_group() as tg,
    ):
        for _ in range(2):
            tg.start_soon(client.call_tool, "list_workspaces", {})

    assert peak == 1


# ---------------------------------------------------------------------------
# Identity and tenancy
# ---------------------------------------------------------------------------


def _workspace_ids(result: CallToolResult) -> list[str]:
    return [ws["id"] for ws in _tool_json(result)["workspaces"]]


async def test_each_request_authorized_by_its_own_token(
    app: FastAPI, alice: Member, bob: Member
) -> None:
    # F1: with sessions, tools saw the user who opened the session forever. One
    # client switching tokens between calls must be seen as each caller in turn.
    auth = _SwappableBearer(alice.token)
    async with running(app), mcp_client(app, auth) as client:
        first = await client.call_tool("list_workspaces", {})
        auth.token = bob.token
        result = await client.call_tool("list_workspaces", {})

    assert _workspace_ids(first) == [str(alice.workspace_id)]
    assert _workspace_ids(result) == [str(bob.workspace_id)]


async def test_two_users_isolated(app: FastAPI, alice: Member, bob: Member) -> None:
    async with (
        running(app),
        mcp_client(app, _SwappableBearer(alice.token)) as as_alice,
        mcp_client(app, _SwappableBearer(bob.token)) as as_bob,
    ):
        seen = [
            _workspace_ids(await c.call_tool("list_workspaces", {}))
            for c in (as_alice, as_bob, as_alice, as_bob)
        ]

    alice_ws, bob_ws = [str(alice.workspace_id)], [str(bob.workspace_id)]
    assert seen == [alice_ws, bob_ws, alice_ws, bob_ws]


async def test_concurrent_callers_each_see_themselves(
    alice: Member, bob: Member, mcp_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both requests pass auth before either tool reads its user, so identity kept
    # anywhere the two requests share (a module global, a shared dict) would show
    # both tools the same user. Sequential calls can't catch that.
    list_workspaces = src.mcp_server.tools.list_workspaces
    arrived = 0
    both_arrived = anyio.Event()

    @functools.wraps(list_workspaces)
    async def _after_both_arrive(*args: Any, **kwargs: Any) -> Any:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await both_arrived.wait()
        return await list_workspaces(*args, **kwargs)

    monkeypatch.setattr(src.mcp_server.tools, "list_workspaces", _after_both_arrive)
    app = create_app()  # registers the patched tool
    results: dict[str, CallToolResult] = {}

    async def _call(name: str, client: Client) -> None:
        results[name] = await client.call_tool("list_workspaces", {})

    async with (
        running(app),
        mcp_client(app, _SwappableBearer(alice.token)) as as_alice,
        mcp_client(app, _SwappableBearer(bob.token)) as as_bob,
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(_call, "alice", as_alice)
        tg.start_soon(_call, "bob", as_bob)

    assert _workspace_ids(results["alice"]) == [str(alice.workspace_id)]
    assert _workspace_ids(results["bob"]) == [str(bob.workspace_id)]


async def test_user_never_written_into_server_state(
    app: FastAPI, alice: Member
) -> None:
    # A server puts its own ``state`` dict in each request's scope (lifespan state,
    # copied per request by uvicorn). The user must not be written into it: if that
    # dict were ever shared, the user would leak into other requests.
    server_state: dict[str, Any] = {"from_lifespan": "kept"}

    async def _server(scope: Scope, receive: Receive, send: Send) -> None:
        scope["state"] = server_state  # the same dict for every request
        await app(scope, receive, send)

    async with running(app), mcp_client(_server, _SwappableBearer(alice.token)) as c:
        result = await c.call_tool("list_workspaces", {})

    assert _workspace_ids(result) == [str(alice.workspace_id)]
    assert server_state == {"from_lifespan": "kept"}


async def test_non_member_workspace_is_tool_error(
    app: FastAPI, alice: Member, bob: Member
) -> None:
    async with running(app), mcp_client(app, _SwappableBearer(alice.token)) as client:
        # Returns rather than raising: the model gets a tool error, not a protocol one.
        result = await client.call_tool(
            "search_documents",
            {"workspace_id": str(bob.workspace_id), "query": "x"},
        )

    assert result.is_error
    assert "Not a member" in result.content[0].text  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Protocol version: the SDK serves pre-2026 clients on a stateful legacy path
# when the header is missing or handshake-era; we refuse them before it.
# ---------------------------------------------------------------------------

# What a pre-2026 client sends first: the handshake, no version header.
LEGACY_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "legacy", "version": "1"},
    },
}


async def _raw(
    app: FastAPI, method: str, headers: dict[str, str], body: object = None
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=ASGITransport(app), base_url=LOCAL_URL, follow_redirects=False
    ) as http:
        return await http.request(
            method,
            "/mcp",
            json=body,
            headers={"Accept": "application/json, text/event-stream", **headers},
        )


@pytest.mark.parametrize("authenticated", [True, False], ids=["token", "no-token"])
async def test_post_without_version_header_is_header_mismatch(
    app: FastAPI, alice: Member, authenticated: bool
) -> None:
    # Spec 2026-07-28: every POST MUST carry MCP-Protocol-Version; a server without
    # header-less clients MUST reject with HeaderMismatch. Refused before auth, and
    # a valid token doesn't open a legacy session either.
    headers = {"Authorization": f"Bearer {alice.token}"} if authenticated else {}
    async with running(app):
        response = await _raw(app, "POST", headers, LEGACY_INITIALIZE)

    assert response.status_code == 400
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": HEADER_MISMATCH, "message": ANY},
    }
    assert "mcp-session-id" not in response.headers


@pytest.mark.parametrize("version", HANDSHAKE_PROTOCOL_VERSIONS)
async def test_handshake_era_version_is_unsupported(
    app: FastAPI, alice: Member, version: str
) -> None:
    # A version the server chose not to support: 400 listing what it does support,
    # so a client that negotiates can retry with it.
    async with running(app):
        response = await _raw(
            app,
            "POST",
            {"Authorization": f"Bearer {alice.token}", "MCP-Protocol-Version": version},
            LEGACY_INITIALIZE,
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert error["data"] == {"supported": [PROTOCOL_VERSION], "requested": version}
    assert "mcp-session-id" not in response.headers


async def test_unknown_newer_version_is_left_to_the_sdk(
    app: FastAPI, alice: Member
) -> None:
    # The guard refuses only handshake-era versions, so a future revision reaches
    # the SDK, which serves it once upgraded. Today the SDK rejects it itself: its
    # error echoes the request id, which the guard (never reading the body) can't.
    future = "2099-01-01"
    body = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/list",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": future,
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    }
    async with running(app):
        response = await _raw(
            app,
            "POST",
            {
                "Authorization": f"Bearer {alice.token}",
                "MCP-Protocol-Version": future,
                "MCP-Method": "tools/list",
            },
            body,
        )

    assert response.status_code == 400
    reply = response.json()
    assert reply["id"] == 7
    assert reply["error"]["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert reply["error"]["data"]["requested"] == future


@pytest.mark.parametrize("version", [None, *HANDSHAKE_PROTOCOL_VERSIONS])
@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT"])
async def test_legacy_non_post_is_405(
    app: FastAPI, alice: Member, method: str, version: str | None
) -> None:
    # A pre-2026 client's GET stream or session DELETE, with or without its
    # handshake-era version: the spec's answer for a 2026-only server, as the SDK
    # gives it when a 2026 version is present.
    headers = {"Authorization": f"Bearer {alice.token}"}
    if version is not None:
        headers["MCP-Protocol-Version"] = version
    async with running(app):
        response = await _raw(app, method, headers)

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


# ---------------------------------------------------------------------------
# Transport security and auth responses
# ---------------------------------------------------------------------------


async def test_non_localhost_host_accepted(app: FastAPI, alice: Member) -> None:
    async with (
        running(app),
        mcp_client(app, _SwappableBearer(alice.token), PUBLIC_URL) as client,
    ):
        result = await client.call_tool("list_workspaces", {})

    assert _workspace_ids(result) == [str(alice.workspace_id)]


@pytest.mark.parametrize("scheme", ["bearer", "BEARER"])
async def test_auth_scheme_is_case_insensitive(
    app: FastAPI, alice: Member, scheme: str
) -> None:
    # RFC 7235 §2.1: the scheme name is case-insensitive; REST accepts it too.
    async with running(app):
        response = await _post_rpc(app, {"Authorization": f"{scheme} {alice.token}"})

    assert response.status_code == 200


NO_CREDENTIALS = 'Bearer realm="knowledge-platform"'
INVALID_TOKEN = 'Bearer realm="knowledge-platform", error="invalid_token"'


def _expired_access_token(user_id: uuid.UUID) -> str:
    now = int(time.time())
    payload = {"sub": str(user_id), "type": "access", "iat": now - 120, "exp": now - 60}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def _inactive_users_token(db_session: AsyncSession) -> str:
    member = await _create_member(db_session, "carol")
    user_id = decode_token(member.token).sub
    await db_session.execute(
        update(User).where(User.id == user_id).values(is_active=False)
    )
    return member.token


@pytest.mark.parametrize(
    ("authorization", "challenge"),
    [
        pytest.param(None, NO_CREDENTIALS, id="missing"),
        pytest.param("Basic YWxpY2U6cHc=", NO_CREDENTIALS, id="not-bearer"),
        pytest.param("Bearer", NO_CREDENTIALS, id="scheme-without-token"),
        pytest.param("Bearer no-such-api-key", INVALID_TOKEN, id="unknown-api-key"),
        pytest.param("Bearer not.a.jwt", INVALID_TOKEN, id="garbage-jwt"),
        pytest.param("expired", INVALID_TOKEN, id="expired-jwt"),
        pytest.param("refresh", INVALID_TOKEN, id="refresh-as-access"),
        pytest.param("inactive", INVALID_TOKEN, id="inactive-user"),
    ],
)
async def test_credential_failures_are_401(
    app: FastAPI,
    db_session: AsyncSession,
    alice: Member,
    authorization: str | None,
    challenge: str,
) -> None:
    # 401, never 403: clients re-authenticate only on 401 (RFC 6750 §3).
    # Tokens name a real, active user, so each fails for its own reason and not
    # because the user can't be found.
    alice_id = decode_token(alice.token).sub
    tokens = {
        "expired": lambda: _expired_access_token(alice_id),
        "refresh": lambda: create_refresh_token(alice_id),
    }
    if authorization == "inactive":
        authorization = f"Bearer {await _inactive_users_token(db_session)}"
    elif authorization in tokens:
        authorization = f"Bearer {tokens[authorization]()}"
    headers = {"Authorization": authorization} if authorization else {}

    response = await _post_rpc(app, headers)

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == challenge
    assert response.json()["error"]["code"] == 401


async def test_auth_backend_failure_is_503_not_401(
    app: FastAPI, alice: Member, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An outage must not look like bad credentials: a 401 would make clients drop a
    # valid token and re-authenticate.
    @asynccontextmanager
    async def _db_down() -> AsyncGenerator[AsyncSession]:
        raise ConnectionRefusedError("database unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(src.composition, "get_session", _db_down)

    response = await _post_rpc(app, {"Authorization": f"Bearer {alice.token}"})

    assert response.status_code == 503
    assert "www-authenticate" not in response.headers


# Any Origin is refused, not just foreign ones: none of our clients is a browser.
# A same-host origin is what a rebinding page presents once the hostname resolves
# to us, and "null" is what sandboxed frames and file:// pages send.
ANY_ORIGIN = [
    pytest.param("https://evil.example", PUBLIC_URL, id="foreign"),
    pytest.param("http://kp.example.com", PUBLIC_URL, id="same-host"),
    pytest.param("http://localhost:8000", LOCAL_URL, id="same-host-localhost"),
    pytest.param("null", PUBLIC_URL, id="null"),
]


@pytest.mark.parametrize(("origin", "base_url"), ANY_ORIGIN)
async def test_any_origin_rejected(
    app: FastAPI, alice: Member, origin: str, base_url: str
) -> None:
    # Refused even with valid credentials: the Origin check doesn't depend on auth.
    async with running(app):
        response = await _post_rpc(
            app,
            {"Authorization": f"Bearer {alice.token}", "Origin": origin},
            base_url,
        )

    assert response.status_code == 403
    assert "Origin" in response.json()["error"]["message"]


@pytest.mark.parametrize(("origin", "base_url"), ANY_ORIGIN)
async def test_origin_checked_before_auth(
    app: FastAPI, origin: str, base_url: str
) -> None:
    # Without a token, auth would answer 401; a 403 shows the Origin check ran first.
    # It must: the rebinding attack needs no credentials, so auth can't be the guard.
    response = await _post_rpc(app, {"Origin": origin}, base_url)

    assert response.status_code == 403
    assert "Origin" in response.json()["error"]["message"]


@pytest.mark.parametrize("method", ["POST", "GET"])
async def test_origin_checked_before_protocol_version(
    app: FastAPI, alice: Member, method: str
) -> None:
    # Spec: an invalid Origin MUST get 403, so it is checked before anything else
    # can answer — here the legacy-protocol guard's 400 (POST) or 405 (GET).
    response = await _raw(
        app,
        method,
        {"Authorization": f"Bearer {alice.token}", "Origin": "https://evil.example"},
        LEGACY_INITIALIZE if method == "POST" else None,
    )

    assert response.status_code == 403
    assert "Origin" in response.json()["error"]["message"]


async def test_sdk_crash_is_the_apps_json_500(app: FastAPI, alice: Member) -> None:
    # /mcp serves the session manager's bare handler, so an SDK crash reaches the
    # app's handler like any REST crash. Crash: the manager is never started.
    transport = ASGITransport(app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url=LOCAL_URL) as http:
        response = await http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={
                "Authorization": f"Bearer {alice.token}",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": 500, "message": "Internal server error", "errors": None}
    }


async def test_unmatched_rest_path_is_404(app: FastAPI) -> None:
    # MCP's auth must not answer for paths outside /mcp.
    async with httpx.AsyncClient(
        transport=ASGITransport(app), base_url=LOCAL_URL
    ) as http:
        response = await http.get("/api/v1/does-not-exist")

    assert response.status_code == 404
