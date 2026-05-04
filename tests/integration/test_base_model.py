"""Verify Base model server-generated fields: uuidv7 PK, created_at, repr."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.workspace import Workspace


async def test_uuidv7_primary_key_is_generated(db_session: AsyncSession) -> None:
    ws = Workspace(name="ACME Corp", slug="acme-corp")
    db_session.add(ws)
    await db_session.flush()

    assert ws.id is not None
    assert isinstance(ws.id, uuid.UUID)
    assert ws.id.version == 7


async def test_created_at_set_by_server(db_session: AsyncSession) -> None:
    ws = Workspace(name="Beta Corp", slug="beta-corp")
    db_session.add(ws)
    await db_session.flush()

    assert ws.created_at is not None


async def test_base_repr(db_session: AsyncSession) -> None:
    ws = Workspace(name="Repr Corp", slug="repr-corp")
    db_session.add(ws)
    await db_session.flush()

    assert repr(ws) == f"<Workspace id={ws.id}>"
