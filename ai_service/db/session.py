from typing import AsyncIterator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from ai_service.config.settings import settings

# Create async engine with robust pool configuration
engine = create_async_engine(
    settings.async_database_url,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

# Async session maker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_db() -> AsyncIterator[AsyncSession]:
    """
    Dependency to yield an async database session.
    Automatically handles rollback on error and closing at request end.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
