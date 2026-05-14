from collections.abc import Generator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

# Celery workers run in synchronous processes and cannot share the asyncpg engine
# used by FastAPI. asyncpg requires an active event loop; mixing sync and async DB
# access in the same process causes pool exhaustion and subtle deadlocks.
# This engine uses psycopg3 (synchronous) and is intentionally separate.
sync_engine = sa.create_engine(
    settings.SYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

sync_session_factory: sessionmaker[Session] = sessionmaker(
    sync_engine,
    expire_on_commit=False,
)


@contextmanager
def get_sync_session() -> Generator[Session]:
    session = sync_session_factory()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
