from enum import StrEnum


class WorkspaceRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


PERMISSIONS: dict[WorkspaceRole, set[str]] = {
    WorkspaceRole.OWNER: {
        "delete_workspace",
        "manage_members",
        "create_document",
        "update_document",
        "delete_document",
        "read",
    },
    WorkspaceRole.ADMIN: {
        "manage_members",
        "create_document",
        "update_document",
        "delete_document",
        "read",
    },
    WorkspaceRole.MEMBER: {"create_document", "update_document", "read"},
    WorkspaceRole.VIEWER: {"read"},
}
