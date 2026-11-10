"""Committed rows for tests that run on real commits (``committing_engine``).

Written through a unit of work, the way the services write (ADR-033).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.documents import ContentType
from src.domain.roles import WorkspaceRole
from src.domain.user import UserCreationInput
from src.models.document import Document
from src.models.user import User
from src.repositories.unit_of_work import unit_of_work_factory


async def user(engine: AsyncEngine, hashed_password: str = "!") -> User:
    async with unit_of_work_factory(engine)() as uow:
        created = await uow.users.create(
            UserCreationInput(email=f"{uuid.uuid4().hex[:12]}@example.com"),
            hashed_password,
        )
        await uow.commit()
    return created


async def workspace_of(engine: AsyncEngine, *members: User) -> uuid.UUID:
    """A workspace owned by the first member; the others join as members."""
    async with unit_of_work_factory(engine)() as uow:
        ws = await uow.workspaces.create(
            name="ws", slug=f"ws-{uuid.uuid4().hex[:8]}", created_by_id=members[0].id
        )
        for i, member in enumerate(members):
            role = WorkspaceRole.OWNER if i == 0 else WorkspaceRole.MEMBER
            await uow.workspaces.add_member(ws.id, member.id, role)
        await uow.commit()
    return ws.id


async def document(
    engine: AsyncEngine, workspace_id: uuid.UUID, uploader: User, **columns: Any
) -> Document:
    """A document row; ``columns`` (status, title, …) are set before the commit."""
    async with unit_of_work_factory(engine)() as uow:
        doc = await uow.documents.create(
            workspace_id=workspace_id,
            title=columns.pop("title", f"doc-{uuid.uuid4().hex[:6]}"),
            content_type=columns.pop("content_type", ContentType.MARKDOWN),
            file_path="",
            file_size_bytes=1,
            uploaded_by=uploader.id,
            original_filename=None,
            trace_context=None,
        )
        for name, value in columns.items():
            setattr(doc, name, value)
        await uow.commit()
    return doc
