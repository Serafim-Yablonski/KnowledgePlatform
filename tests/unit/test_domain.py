"""Unit tests for src/domain/."""

import base64
import json
import uuid
from datetime import datetime

import pytest

from src.domain.documents import Cursor, decode_cursor, encode_cursor
from src.domain.roles import PERMISSIONS, WorkspaceRole


def test_workspace_role_values() -> None:
    assert WorkspaceRole.OWNER == "owner"
    assert WorkspaceRole.ADMIN == "admin"
    assert WorkspaceRole.MEMBER == "member"
    assert WorkspaceRole.VIEWER == "viewer"


def test_permissions_coverage() -> None:
    assert "delete_workspace" in PERMISSIONS[WorkspaceRole.OWNER]
    assert "manage_members" in PERMISSIONS[WorkspaceRole.ADMIN]
    assert "create_document" in PERMISSIONS[WorkspaceRole.MEMBER]
    assert PERMISSIONS[WorkspaceRole.VIEWER] == {"read"}


def test_viewer_cannot_write() -> None:
    assert "create_document" not in PERMISSIONS[WorkspaceRole.VIEWER]
    assert "delete_workspace" not in PERMISSIONS[WorkspaceRole.VIEWER]


def test_owner_is_superset_of_admin() -> None:
    assert PERMISSIONS[WorkspaceRole.ADMIN].issubset(PERMISSIONS[WorkspaceRole.OWNER])


def test_all_roles_have_read() -> None:
    for role in WorkspaceRole:
        assert "read" in PERMISSIONS[role]


def _b64(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


_NAIVE = "2026-10-22T12:00:00.123456"
_ID = "6f9619ff-8b86-d011-b42d-00cf4fc964ff"
_AWARE = _NAIVE + "+00:00"


class TestPaginationCursor:
    def test_round_trips_what_the_repository_produces(self) -> None:
        # created_at is a timezone-naive column, so the repository's cursors are naive.
        cursor = Cursor(created_at=datetime(2026, 10, 22, 12, 0, 0, 1), id=uuid.uuid4())
        assert decode_cursor(encode_cursor(cursor)) == cursor

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(_b64([_NAIVE, _ID]), id="json-list"),
            pytest.param(_b64(7), id="json-int"),
            pytest.param(_b64(None), id="json-null"),
            pytest.param(_b64({"created_at": _NAIVE}), id="missing-id"),
            pytest.param(_b64({"created_at": 0, "id": _ID}), id="numeric-created-at"),
            pytest.param(_b64({"created_at": _NAIVE, "id": 7}), id="numeric-id"),
            # Our cursors are never aware; asyncpg rejects one against the naive column.
            pytest.param(
                _b64({"created_at": _NAIVE + "+00:00", "id": _ID}), id="tz-aware"
            ),
            pytest.param(
                _b64({"created_at": _NAIVE, "id": _ID, "x": 1}), id="extra-key"
            ),
            pytest.param(base64.urlsafe_b64encode(b"\xff\xfe").decode(), id="not-utf8"),
            pytest.param("not-valid-base64!!", id="not-base64"),
            pytest.param("A" * 513, id="too-long"),
        ],
    )
    def test_anything_but_our_own_shape_is_a_value_error(self, value: str) -> None:
        with pytest.raises(ValueError, match="Invalid pagination cursor"):
            decode_cursor(value)
