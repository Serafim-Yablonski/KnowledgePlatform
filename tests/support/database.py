"""Scratch databases in the session's PostgreSQL container."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from alembic.config import Config
from testcontainers.postgres import PostgresContainer

from alembic import command


@contextmanager
def scratch_database(container: PostgresContainer, prefix: str) -> Iterator[str]:
    """A fresh, empty database, dropped on exit; yields its psycopg URL."""
    base = sa.make_url(container.get_connection_url())
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(
        base.set(drivername="postgresql+psycopg"), isolation_level="AUTOCOMMIT"
    )
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    try:
        yield base.set(drivername="postgresql+psycopg", database=name).render_as_string(
            hide_password=False
        )
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def alembic_config(url: str) -> Config:
    """Alembic aimed at ``url`` (env.py runs on asyncpg). Call ``command`` with it
    outside any running event loop: env.py calls asyncio.run()."""
    cfg = Config("alembic.ini")
    cfg.set_main_option(
        "sqlalchemy.url", url.replace("postgresql+psycopg", "postgresql+asyncpg")
    )
    return cfg


def migrate(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)
