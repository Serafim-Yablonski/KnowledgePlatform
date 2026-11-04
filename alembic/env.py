import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from src.core.config import settings
from src.models import Base

alembic_config = context.config

if alembic_config.config_file_name is not None:
    # The default would disable every logger that already exists — all of them when
    # migrations run inside another process (the test suite), not via the CLI.
    fileConfig(alembic_config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _get_url() -> str:
    # Prefer URL set programmatically (e.g. from test fixtures) over settings.
    url = alembic_config.get_main_option("sqlalchemy.url")
    return url if url else settings.ASYNC_DATABASE_URL.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Server defaults too, as the drift test compares them: a changed default in a
    # model then shows up in `make revision`, not only as a failing test.
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(_get_url(), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
