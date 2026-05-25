"""Unit tests for MCP session tools and require_active_workspace helper."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ForbiddenError
from src.mcp_server.auth import _current_user
from src.mcp_server.session import (
    McpSessionState,
    _current_session_id,
    _sessions,
    get_or_create_session_state,
    require_active_workspace,
)
from src.mcp_server.tools import list_workspaces, set_active_workspace
from src.schemas.workspace import WorkspaceResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.is_active = True
    return user


def _make_workspace_response(ws_id: uuid.UUID | None = None) -> WorkspaceResponse:
    return WorkspaceResponse(
        id=ws_id or uuid.uuid4(),
        name="Test Workspace",
        slug="test-workspace",
        description=None,
        is_active=True,
        created_at=__import__("datetime").datetime(2026, 1, 1),
        member_count=3,
    )


@pytest.fixture
def mock_user():
    user = _make_user()
    token = _current_user.set(user)
    yield user
    _current_user.reset(token)


@pytest.fixture
def session_id() -> str:
    sid = f"test-session-{uuid.uuid4().hex}"
    yield sid
    _sessions.pop(sid, None)


@pytest.fixture
def mock_session_ctx():
    session = MagicMock()

    @asynccontextmanager
    async def _fake() -> AsyncGenerator[MagicMock]:
        yield session

    with patch("src.mcp_server.tools.get_session", _fake):
        yield session


@pytest.fixture
def mock_redis():
    with patch("src.mcp_server.tools.get_async_redis_client", return_value=MagicMock()):
        yield


# ---------------------------------------------------------------------------
# list_workspaces
# ---------------------------------------------------------------------------


class TestListWorkspaces:
    @pytest.mark.asyncio
    async def test_returns_user_workspaces(
        self, mock_user: MagicMock, session_id: str, mock_session_ctx: MagicMock
    ) -> None:
        ws_id = uuid.uuid4()
        ws = _make_workspace_response(ws_id)
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.list_for_user = AsyncMock(return_value=[ws])
        mock_workspace_svc.get_user_role = AsyncMock(return_value="member")

        sid_token = _current_session_id.set(session_id)
        try:
            with patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ):
                result = await list_workspaces()
        finally:
            _current_session_id.reset(sid_token)

        assert len(result) == 1
        assert result[0]["id"] == str(ws_id)
        assert result[0]["name"] == "Test Workspace"
        assert result[0]["slug"] == "test-workspace"
        assert "role" in result[0]

    @pytest.mark.asyncio
    async def test_creates_session_state_entry(
        self, mock_user: MagicMock, session_id: str, mock_session_ctx: MagicMock
    ) -> None:
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.list_for_user = AsyncMock(return_value=[])
        mock_workspace_svc.get_user_role = AsyncMock(return_value="member")

        sid_token = _current_session_id.set(session_id)
        try:
            with patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ):
                await list_workspaces()
        finally:
            _current_session_id.reset(sid_token)

        assert session_id in _sessions


# ---------------------------------------------------------------------------
# set_active_workspace
# ---------------------------------------------------------------------------


class TestSetActiveWorkspace:
    @pytest.mark.asyncio
    async def test_stores_workspace_in_session_state(
        self, mock_user: MagicMock, session_id: str, mock_session_ctx: MagicMock
    ) -> None:
        from src.domain.roles import WorkspaceRole

        ws_id = uuid.uuid4()
        ws = _make_workspace_response(ws_id)
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(return_value=WorkspaceRole.MEMBER)
        mock_workspace_svc.get_by_id = AsyncMock(return_value=ws)

        get_or_create_session_state(session_id, mock_user)
        sid_token = _current_session_id.set(session_id)
        try:
            with patch(
                "src.mcp_server.tools._make_workspace_service",
                return_value=mock_workspace_svc,
            ):
                result = await set_active_workspace(workspace_id=str(ws_id))
        finally:
            _current_session_id.reset(sid_token)

        assert result["active_workspace_id"] == str(ws_id)
        assert result["name"] == "Test Workspace"
        state = _sessions[session_id]
        assert state.active_workspace_id == ws_id
        assert state.active_workspace_name == "Test Workspace"
        assert state.active_workspace_role == WorkspaceRole.MEMBER

    @pytest.mark.asyncio
    async def test_non_member_raises_forbidden(
        self, mock_user: MagicMock, session_id: str, mock_session_ctx: MagicMock
    ) -> None:
        mock_workspace_svc = MagicMock()
        mock_workspace_svc.get_user_role = AsyncMock(
            side_effect=ForbiddenError("Not a member")
        )

        get_or_create_session_state(session_id, mock_user)
        sid_token = _current_session_id.set(session_id)
        try:
            with (
                patch(
                    "src.mcp_server.tools._make_workspace_service",
                    return_value=mock_workspace_svc,
                ),
                pytest.raises(ForbiddenError),
            ):
                await set_active_workspace(workspace_id=str(uuid.uuid4()))
        finally:
            _current_session_id.reset(sid_token)


# ---------------------------------------------------------------------------
# require_active_workspace
# ---------------------------------------------------------------------------


class TestRequireActiveWorkspace:
    def test_raises_when_no_workspace_set(self, session_id: str) -> None:
        user = _make_user()
        get_or_create_session_state(session_id, user)

        from mcp import McpError

        with pytest.raises(McpError, match="No active workspace"):
            require_active_workspace(session_id)

    def test_returns_uuid_when_workspace_set(self, session_id: str) -> None:
        from src.domain.roles import WorkspaceRole

        ws_id = uuid.uuid4()
        user = _make_user()
        state = get_or_create_session_state(session_id, user)
        state.active_workspace_id = ws_id
        state.active_workspace_name = "Ws"
        state.active_workspace_role = WorkspaceRole.MEMBER

        result = require_active_workspace(session_id)
        assert result == ws_id

    def test_session_state_has_correct_user(self, session_id: str) -> None:
        user = _make_user()
        state = McpSessionState(user=user)
        _sessions[session_id] = state
        assert _sessions[session_id].user is user
