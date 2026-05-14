import uuid
from typing import Protocol

from src.domain.documents import ContentType, Cursor, DocumentStatus
from src.domain.roles import WorkspaceRole
from src.models.document import Document
from src.models.user import User
from src.models.workspace import Workspace, WorkspaceMembership
from src.schemas.auth import UserCreate
from src.schemas.document import DocumentUpdate


class UserRepositoryProtocol(Protocol):
    async def get_by_id(self, user_id: uuid.UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def create(self, data: UserCreate, hashed_password: str) -> User: ...

    async def update(self, user: User, *, is_active: bool | None = None) -> User: ...

    async def exists_by_email(self, email: str) -> bool: ...


class WorkspaceRepositoryProtocol(Protocol):
    async def create(
        self,
        name: str,
        slug: str,
        created_by_id: uuid.UUID,
        description: str | None = None,
    ) -> Workspace: ...

    async def get_by_id(self, workspace_id: uuid.UUID) -> Workspace | None: ...

    async def get_by_slug(self, slug: str) -> Workspace | None: ...

    async def list_for_user(self, user_id: uuid.UUID) -> list[Workspace]: ...

    async def add_member(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: WorkspaceRole,
        invited_by_id: uuid.UUID | None = None,
    ) -> WorkspaceMembership: ...

    async def remove_member(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> None: ...

    async def get_membership(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMembership | None: ...

    async def list_members(
        self, workspace_id: uuid.UUID
    ) -> list[WorkspaceMembership]: ...

    async def count_members(self, workspace_id: uuid.UUID) -> int: ...


class DocumentRepositoryProtocol(Protocol):
    async def create(
        self,
        workspace_id: uuid.UUID,
        title: str,
        content_type: ContentType,
        file_path: str,
        file_size_bytes: int,
        uploaded_by: uuid.UUID,
    ) -> Document: ...

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None: ...

    async def update(self, document: Document, data: DocumentUpdate) -> Document: ...

    async def delete(self, document: Document) -> None: ...

    async def list_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        limit: int = 20,
        cursor: Cursor | None = None,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], Cursor | None]: ...
