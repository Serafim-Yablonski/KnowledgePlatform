"""Unit tests for src/domain/."""

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
