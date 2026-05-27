"""Unit tests for WorkspaceService using in-memory Protocol stubs."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from src.domain.roles import WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace, WorkspaceMembership
from src.services.workspace import WorkspaceService

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_user(email: str = "user@example.com") -> User:
    u = User()
    u.id = uuid.uuid4()
    u.email = email
    u.hashed_password = "hashed"
    u.display_name = None
    u.is_active = True
    u.created_at = datetime.now(UTC).replace(tzinfo=None)
    u.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return u


def _make_workspace(created_by: uuid.UUID) -> Workspace:
    ws = Workspace()
    ws.id = uuid.uuid4()
    ws.name = "Test WS"
    ws.slug = "test-ws-abcd"
    ws.description = None
    ws.created_by = created_by
    ws.is_active = True
    ws.created_at = datetime.now(UTC).replace(tzinfo=None)
    ws.updated_at = datetime.now(UTC).replace(tzinfo=None)
    return ws


def _make_membership(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    role: WorkspaceRole,
    user: User | None = None,
) -> WorkspaceMembership:
    m = WorkspaceMembership()
    m.workspace_id = workspace_id
    m.user_id = user_id
    m.role = role
    m.invited_by = None
    m.joined_at = datetime.now(UTC).replace(tzinfo=None)
    # Populate the relationship so service.list_members() can access m.user.*
    # without hitting the database (lazy="raise" would fire on a real session).
    m.user = user or _make_user()  # type: ignore[assignment]
    return m


class StubWorkspaceRepository:
    def __init__(self, user_store: dict[uuid.UUID, User] | None = None) -> None:
        self._workspaces: dict[uuid.UUID, Workspace] = {}
        self._memberships: list[WorkspaceMembership] = []
        # Allows list_members to populate m.user without a real DB session.
        self._user_store: dict[uuid.UUID, User] = user_store or {}

    async def create(
        self,
        name: str,
        slug: str,
        created_by_id: uuid.UUID,
        description: str | None = None,
    ) -> Workspace:
        ws = Workspace()
        ws.id = uuid.uuid4()
        ws.name = name
        ws.slug = slug
        ws.description = description
        ws.created_by = created_by_id
        ws.is_active = True
        ws.created_at = datetime.now(UTC).replace(tzinfo=None)
        ws.updated_at = datetime.now(UTC).replace(tzinfo=None)
        self._workspaces[ws.id] = ws
        return ws

    async def get_by_id(self, workspace_id: uuid.UUID) -> Workspace | None:
        return self._workspaces.get(workspace_id)

    async def get_by_slug(self, slug: str) -> Workspace | None:
        return next((w for w in self._workspaces.values() if w.slug == slug), None)

    async def list_for_user(self, user_id: uuid.UUID) -> list[Workspace]:
        member_ws_ids = {
            m.workspace_id for m in self._memberships if m.user_id == user_id
        }
        return [w for w in self._workspaces.values() if w.id in member_ws_ids]

    async def add_member(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: WorkspaceRole,
        invited_by_id: uuid.UUID | None = None,
    ) -> WorkspaceMembership:
        user = self._user_store.get(user_id)
        m = _make_membership(workspace_id, user_id, role, user=user)
        m.invited_by = invited_by_id
        self._memberships.append(m)
        return m

    async def remove_member(self, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
        self._memberships = [
            m
            for m in self._memberships
            if not (m.workspace_id == workspace_id and m.user_id == user_id)
        ]

    async def get_membership(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMembership | None:
        return next(
            (
                m
                for m in self._memberships
                if m.workspace_id == workspace_id and m.user_id == user_id
            ),
            None,
        )

    async def list_members(self, workspace_id: uuid.UUID) -> list[WorkspaceMembership]:
        return [m for m in self._memberships if m.workspace_id == workspace_id]

    async def count_members(self, workspace_id: uuid.UUID) -> int:
        return sum(1 for m in self._memberships if m.workspace_id == workspace_id)

    async def count_owners_for_update(self, workspace_id: uuid.UUID) -> int:
        return sum(
            1
            for m in self._memberships
            if m.workspace_id == workspace_id and m.role == WorkspaceRole.OWNER
        )


class StubUserRepository:
    def __init__(self, users: list[User] | None = None) -> None:
        self._store: dict[uuid.UUID, User] = {u.id: u for u in (users or [])}

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return self._store.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        return next((u for u in self._store.values() if u.email == email), None)

    async def create(self, data: object, hashed_password: str) -> User:  # type: ignore[override]
        raise NotImplementedError

    async def update(self, user: User, *, is_active: bool | None = None) -> User:
        raise NotImplementedError

    async def exists_by_email(self, email: str) -> bool:
        return any(u.email == email for u in self._store.values())


def _make_service(
    extra_users: list[User] | None = None,
) -> tuple[WorkspaceService, StubWorkspaceRepository, StubUserRepository]:
    users = extra_users or []
    user_store = {u.id: u for u in users}
    ws_repo = StubWorkspaceRepository(user_store=user_store)
    user_repo = StubUserRepository(users)
    return WorkspaceService(ws_repo, user_repo), ws_repo, user_repo


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


async def test_create_adds_owner_membership() -> None:
    service, ws_repo, _ = _make_service()
    owner = _make_user()
    result = await service.create(owner, "My Team")

    assert result.name == "My Team"
    assert result.member_count == 1

    membership = await ws_repo.get_membership(result.id, owner.id)
    assert membership is not None
    assert membership.role == WorkspaceRole.OWNER


async def test_create_returns_workspace_response_with_slug() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    result = await service.create(owner, "eng hub")
    assert result.slug.startswith("eng-hub-")


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


async def test_member_can_get_workspace() -> None:
    service, ws_repo, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Workspace")
    result = await service.get_by_id(owner, created.id)
    assert result.id == created.id


async def test_non_member_cannot_get_workspace() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    stranger = _make_user("stranger@example.com")
    created = await service.create(owner, "Private")
    with pytest.raises(ForbiddenError):
        await service.get_by_id(stranger, created.id)


# ---------------------------------------------------------------------------
# add_member
# ---------------------------------------------------------------------------


async def test_owner_can_add_member() -> None:
    new_user = _make_user("new@example.com")
    service, _, _ = _make_service(extra_users=[new_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    result = await service.add_member(owner, created.id, "new@example.com")
    assert result.user_id == new_user.id
    assert result.role == WorkspaceRole.MEMBER


async def test_member_cannot_add_members() -> None:
    member_user = _make_user("member@example.com")
    new_user = _make_user("new@example.com")
    service, ws_repo, _ = _make_service(extra_users=[member_user, new_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, member_user.id, WorkspaceRole.MEMBER)

    with pytest.raises(ForbiddenError):
        await service.add_member(member_user, created.id, "new@example.com")


async def test_add_duplicate_member_raises_conflict() -> None:
    existing_user = _make_user("existing@example.com")
    service, ws_repo, _ = _make_service(extra_users=[existing_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, existing_user.id, WorkspaceRole.MEMBER)

    with pytest.raises(ConflictError):
        await service.add_member(owner, created.id, "existing@example.com")


async def test_add_nonexistent_user_returns_same_error_as_duplicate() -> None:
    # Both unknown email and already-member return ConflictError to prevent
    # email-existence enumeration by ADMIN-level callers.
    service, _, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")
    with pytest.raises(ConflictError):
        await service.add_member(owner, created.id, "ghost@example.com")


async def test_admin_cannot_grant_owner_role() -> None:
    admin_user = _make_user("admin@example.com")
    new_user = _make_user("new@example.com")
    service, ws_repo, _ = _make_service(extra_users=[admin_user, new_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, admin_user.id, WorkspaceRole.ADMIN)

    with pytest.raises(ForbiddenError, match="higher than your own"):
        await service.add_member(
            admin_user,
            created.id,
            "new@example.com",
            WorkspaceRole.OWNER,
        )


async def test_admin_cannot_remove_higher_ranked_owner() -> None:
    admin_user = _make_user("admin@example.com")
    service, ws_repo, _ = _make_service(extra_users=[admin_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, admin_user.id, WorkspaceRole.ADMIN)

    with pytest.raises(ForbiddenError, match="higher role"):
        await service.remove_member(admin_user, created.id, owner.id)


# ---------------------------------------------------------------------------
# remove_member
# ---------------------------------------------------------------------------


async def test_cannot_remove_last_owner() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")
    with pytest.raises(ConflictError, match="last owner"):
        await service.remove_member(owner, created.id, owner.id)


async def test_owner_can_remove_regular_member() -> None:
    member_user = _make_user("member@example.com")
    service, ws_repo, _ = _make_service(extra_users=[member_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, member_user.id, WorkspaceRole.MEMBER)

    await service.remove_member(owner, created.id, member_user.id)
    membership = await ws_repo.get_membership(created.id, member_user.id)
    assert membership is None


async def test_member_cannot_remove_others() -> None:
    member_user = _make_user("member@example.com")
    other_user = _make_user("other@example.com")
    service, ws_repo, _ = _make_service(extra_users=[member_user, other_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, member_user.id, WorkspaceRole.MEMBER)
    await ws_repo.add_member(created.id, other_user.id, WorkspaceRole.MEMBER)

    with pytest.raises(ForbiddenError):
        await service.remove_member(member_user, created.id, other_user.id)


async def test_admin_cannot_remove_last_owner() -> None:
    admin_user = _make_user("admin@example.com")
    service, ws_repo, _ = _make_service(extra_users=[admin_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    await ws_repo.add_member(created.id, admin_user.id, WorkspaceRole.ADMIN)

    # Role-rank check fires first: ADMIN cannot remove OWNER at all.
    with pytest.raises(ForbiddenError, match="higher role"):
        await service.remove_member(admin_user, created.id, owner.id)


# ---------------------------------------------------------------------------
# list_members
# ---------------------------------------------------------------------------


async def test_list_members_returns_all_with_user_info() -> None:
    member_user = _make_user("listed@example.com")
    member_user.display_name = "Listed User"
    service, ws_repo, _ = _make_service(extra_users=[member_user])
    owner = _make_user()
    created = await service.create(owner, "Team")
    # ws_repo.add_member uses _user_store to populate m.user automatically
    await ws_repo.add_member(created.id, member_user.id, WorkspaceRole.MEMBER)

    result = await service.list_members(owner, created.id)
    assert len(result) == 2
    emails = {r.email for r in result}
    assert "listed@example.com" in emails


async def test_list_members_non_member_forbidden() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    stranger = _make_user("stranger@example.com")
    created = await service.create(owner, "Team")

    with pytest.raises(ForbiddenError):
        await service.list_members(stranger, created.id)


# ---------------------------------------------------------------------------
# get_workspace_for_user
# ---------------------------------------------------------------------------


async def test_get_workspace_for_user_returns_workspace_and_membership() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Auth WS")

    workspace, membership = await service.get_workspace_for_user(created.id, owner.id)

    assert workspace.id == created.id
    assert membership.user_id == owner.id
    assert membership.role == WorkspaceRole.OWNER


async def test_get_workspace_for_user_non_member_raises_forbidden() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    stranger = _make_user("stranger@example.com")
    created = await service.create(owner, "Private WS")

    with pytest.raises(ForbiddenError):
        await service.get_workspace_for_user(created.id, stranger.id)


async def test_get_workspace_for_user_unknown_workspace_raises_forbidden() -> None:
    service, _, _ = _make_service()
    user = _make_user()

    with pytest.raises(ForbiddenError):
        await service.get_workspace_for_user(uuid.uuid4(), user.id)


async def test_get_by_id_workspace_inconsistency_raises_not_found() -> None:
    """Membership exists but workspace row deleted — get_by_id raises NotFoundError."""
    service, ws_repo, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")
    ws_repo._workspaces.pop(created.id)

    with pytest.raises(NotFoundError):
        await service.get_by_id(owner, created.id)


async def test_get_workspace_for_user_workspace_inconsistency_raises_forbidden() -> (
    None
):
    """Membership exists but workspace deleted raises ForbiddenError."""
    service, ws_repo, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")
    ws_repo._workspaces.pop(created.id)

    with pytest.raises(ForbiddenError):
        await service.get_workspace_for_user(created.id, owner.id)


# ---------------------------------------------------------------------------
# remove_member — not found path
# ---------------------------------------------------------------------------


async def test_remove_member_target_not_found_raises_not_found() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")

    with pytest.raises(NotFoundError):
        await service.remove_member(owner, created.id, uuid.uuid4())


# ---------------------------------------------------------------------------
# add_member — IntegrityError race condition path
# ---------------------------------------------------------------------------


async def test_add_member_integrity_error_raises_conflict() -> None:
    new_user = _make_user("new@example.com")
    service, ws_repo, _ = _make_service(extra_users=[new_user])
    owner = _make_user()
    created = await service.create(owner, "Team")

    async def _raise(*args: object, **kwargs: object) -> WorkspaceMembership:
        raise IntegrityError(None, None, None)

    ws_repo.add_member = _raise  # type: ignore[method-assign]

    with pytest.raises(ConflictError):
        await service.add_member(owner, created.id, "new@example.com")


# ---------------------------------------------------------------------------
# get_user_role
# ---------------------------------------------------------------------------


async def test_get_user_role_returns_role() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    created = await service.create(owner, "Team")

    role = await service.get_user_role(owner, created.id)
    assert role == WorkspaceRole.OWNER


async def test_get_user_role_not_member_raises_forbidden() -> None:
    service, _, _ = _make_service()
    owner = _make_user()
    stranger = _make_user("stranger@example.com")
    created = await service.create(owner, "Team")

    with pytest.raises(ForbiddenError):
        await service.get_user_role(stranger, created.id)
