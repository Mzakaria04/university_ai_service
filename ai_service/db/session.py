from typing import AsyncIterator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from ai_service.config.settings import settings
from ai_service.db.models import AISession, AIMessageEvent, AIFeedback, AIToolExecutionLog, AIExecutionTrace

# Create async engine with robust pool configuration (Write-enabled for AI-owned tables)
engine = create_async_engine(
    settings.async_database_url,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

# Create async engine for read-only core university data queries
import sys
if "pytest" in sys.modules:
    # Reuse the primary engine in test suite execution to allow test fixtures to cleanly
    # dispose of the connection pools and prevent "Event loop is closed" errors.
    readonly_engine = engine
else:
    readonly_engine = create_async_engine(
        settings.async_readonly_database_url,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,
    )

# Async session maker utilizing routing binds
AsyncSessionLocal = async_sessionmaker(
    bind=readonly_engine,  # Default to read-only for raw SQL and university tables
    binds={
        AISession: engine,
        AIMessageEvent: engine,
        AIFeedback: engine,
        AIToolExecutionLog: engine,
        AIExecutionTrace: engine,
    },
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
