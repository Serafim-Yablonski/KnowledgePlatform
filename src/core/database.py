from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def build_engine(url: str) -> AsyncEngine:
    """The API process's engine; its lifespan builds one (ADR-033).

    Sessions come only from the unit of work (src/repositories/unit_of_work.py).
    ``pool_pre_ping``: without it the first request on each stale pooled connection
    after a Postgres restart is a 500.
    """
    return create_async_engine(
        url, pool_size=10, max_overflow=20, pool_pre_ping=True, echo=False
    )
