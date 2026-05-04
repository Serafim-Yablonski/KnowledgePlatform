from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with get_session() as session:
        yield session
