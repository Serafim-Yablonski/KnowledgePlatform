"""Per-session state for the MCP server.

Each MCP session (identified by the mcp-session-id header) holds a
McpSessionState instance in an in-memory dict.  State is intentionally
ephemeral: a server restart clears it and the client calls list_workspaces /
set_active_workspace again.
"""

from __future__ import annotations

import contextvars
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.exceptions import InputValidationError

if TYPE_CHECKING:
    from src.domain.roles import WorkspaceRole
    from src.models.user import User


@dataclass
class McpSessionState:
    user: User
    active_workspace_id: uuid.UUID | None = field(default=None)
    active_workspace_name: str | None = field(default=None)
    active_workspace_role: WorkspaceRole | None = field(default=None)


# Module-level store: session_id → state.  One entry per live MCP session.
# Capped to prevent unbounded memory growth under repeated unauthenticated probes.
_MAX_SESSIONS = 10_000
_sessions: OrderedDict[str, McpSessionState] = OrderedDict()

# Set by MCPAuthMiddleware per HTTP request so tool functions can call
# get_current_session_id() without needing a Context parameter.
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_session_id", default=None
)


def get_current_session_id() -> str | None:
    return _current_session_id.get()


def get_or_create_session_state(session_id: str, user: User) -> McpSessionState:
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
    else:
        if len(_sessions) >= _MAX_SESSIONS:
            # Evict least-recently-used entry (front of the OrderedDict).
            _sessions.popitem(last=False)
        _sessions[session_id] = McpSessionState(user=user)
    return _sessions[session_id]


def get_session_state(session_id: str) -> McpSessionState:
    state = _sessions.get(session_id)
    if state is None:
        raise InputValidationError(
            f"No session state found for session {session_id!r}. "
            "This should not happen — the auth middleware creates state on first use."
        )
    _sessions.move_to_end(session_id)
    return state


def require_active_workspace(session_id: str) -> uuid.UUID:
    """Return active workspace UUID or raise McpError if none has been set."""
    from mcp import McpError
    from mcp.types import ErrorData

    state = get_session_state(session_id)
    if state.active_workspace_id is None:
        raise McpError(
            ErrorData(
                code=-32600,
                message=(
                    "No active workspace. Call list_workspaces then "
                    "set_active_workspace before searching or asking questions."
                ),
            )
        )
    return state.active_workspace_id


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
