from enum import StrEnum

from src.core.exceptions import ForbiddenError


class WorkspaceRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(StrEnum):
    READ = "read"
    CREATE_DOCUMENT = "create_document"
    UPDATE_DOCUMENT = "update_document"
    DELETE_DOCUMENT = "delete_document"
    UPDATE_WORKSPACE = "update_workspace"
    DELETE_WORKSPACE = "delete_workspace"
    MANAGE_MEMBERS = "manage_members"


# Higher rank = more privileges. Used to prevent lower roles from granting or
# revoking higher roles (e.g. ADMIN cannot assign or remove OWNER).
ROLE_RANK: dict[WorkspaceRole, int] = {
    WorkspaceRole.OWNER: 3,
    WorkspaceRole.ADMIN: 2,
    WorkspaceRole.MEMBER: 1,
    WorkspaceRole.VIEWER: 0,
}

PERMISSIONS: dict[WorkspaceRole, set[Permission]] = {
    WorkspaceRole.OWNER: {
        Permission.DELETE_WORKSPACE,
        Permission.UPDATE_WORKSPACE,
        Permission.MANAGE_MEMBERS,
        Permission.CREATE_DOCUMENT,
        Permission.UPDATE_DOCUMENT,
        Permission.DELETE_DOCUMENT,
        Permission.READ,
    },
    WorkspaceRole.ADMIN: {
        Permission.UPDATE_WORKSPACE,
        Permission.MANAGE_MEMBERS,
        Permission.CREATE_DOCUMENT,
        Permission.UPDATE_DOCUMENT,
        Permission.DELETE_DOCUMENT,
        Permission.READ,
    },
    WorkspaceRole.MEMBER: {
        Permission.CREATE_DOCUMENT,
        Permission.UPDATE_DOCUMENT,
        Permission.READ,
    },
    WorkspaceRole.VIEWER: {Permission.READ},
}


def require_permission(role: WorkspaceRole, permission: Permission) -> None:
    if permission not in PERMISSIONS.get(role, set()):
        raise ForbiddenError(f"Insufficient permissions: {permission} required")
