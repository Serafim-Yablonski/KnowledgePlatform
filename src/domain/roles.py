from enum import StrEnum


class WorkspaceRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


# Higher rank = more privileges. Used to prevent lower roles from granting or
# revoking higher roles (e.g. ADMIN cannot assign or remove OWNER).
ROLE_RANK: dict[WorkspaceRole, int] = {
    WorkspaceRole.OWNER: 3,
    WorkspaceRole.ADMIN: 2,
    WorkspaceRole.MEMBER: 1,
    WorkspaceRole.VIEWER: 0,
}

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
