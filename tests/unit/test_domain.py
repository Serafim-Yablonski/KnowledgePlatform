"""Unit tests for src/domain/."""

import base64
import json
import uuid
from datetime import datetime

import pytest

from src.core.exceptions import InputValidationError
from src.domain.documents import (
    Cursor,
    clean_display_text,
    clean_original_filename,
    decode_cursor,
    encode_cursor,
    title_from_filename,
)
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


# ---------------------------------------------------------------------------
# Display text: original filenames and titles
# ---------------------------------------------------------------------------


# Zero-width (non-)joiners are part of real names: an emoji family, a Persian word.
_FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467.pdf"
_PERSIAN = "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645.txt"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("report.pdf", "report.pdf", id="plain"),
        pytest.param("C:\\Users\\me\\a.pdf", "a.pdf", id="windows-path"),
        pytest.param("../../etc/a.pdf", "a.pdf", id="posix-path"),
        pytest.param("re\u0301sume\u0301.pdf", "r\u00e9sum\u00e9.pdf", id="nfd-to-nfc"),
        pytest.param("  report.pdf ", "report.pdf", id="trimmed"),
        pytest.param(_FAMILY, _FAMILY, id="zwj-emoji"),
        pytest.param(_PERSIAN, _PERSIAN, id="zwnj-persian"),
        pytest.param("x" * 251 + ".pdf", "x" * 251 + ".pdf", id="255-chars"),
    ],
)
def test_clean_original_filename_keeps_real_names(raw: str, expected: str) -> None:
    assert clean_original_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", ".", "..", "/", "  ", "dir/"])
def test_clean_original_filename_treats_path_artifacts_as_no_name(raw: str) -> None:
    assert clean_original_filename(raw) is None


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param("a\tb.pdf", "U\\+0009", id="tab"),
        pytest.param("a\x00b.pdf", "U\\+0000", id="nul"),
        pytest.param("a\r\nb.pdf", "U\\+000D", id="crlf"),
        pytest.param("invoice\u202efdp.exe", "U\\+202E", id="rlo"),
        pytest.param("a\u2066b.pdf", "U\\+2066", id="isolate"),
        pytest.param("x" * 252 + ".pdf", "255", id="256-chars"),
    ],
)
def test_clean_original_filename_rejects_unsafe_names(raw: str, match: str) -> None:
    with pytest.raises(InputValidationError, match=match):
        clean_original_filename(raw)


def test_clean_display_text_keeps_slashes_in_titles() -> None:
    assert clean_display_text(" Q3/Q4 report ", "Title") == "Q3/Q4 report"


def test_clean_display_text_names_the_field() -> None:
    with pytest.raises(InputValidationError, match="^Title contains"):
        clean_display_text("a\u202eb", "Title")


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("report-v3.pdf", "report-v3"),
        ("archive.tar.gz", "archive.tar"),
        (".env", ".env"),
        ("README", "README"),
    ],
)
def test_title_from_filename_is_the_stem(name: str, title: str) -> None:
    assert title_from_filename(name) == title
