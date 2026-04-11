"""
Database configuration and session management.

Provides async SQLAlchemy engine and session factory with connection pooling.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, Pool, QueuePool

from bufferiq.core.config import Settings
from bufferiq.domain.base import Base


def get_async_engine(settings: Settings) -> AsyncEngine:
    """
    Create async database engine.
    """

    database_url = settings.database_url

    if database_url.startswith("sqlite://"):
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")

    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    poolclass: type[Pool]
    connect_args: dict[str, Any] = {}
    engine_args: dict[str, Any] = {}

    if "sqlite" in database_url:
        poolclass = NullPool
        connect_args = {"check_same_thread": False}
    else:
        poolclass = QueuePool
        engine_args["pool_size"] = 5
        engine_args["max_overflow"] = 10

    engine = create_async_engine(
        database_url,
        echo=settings.database_echo,
        poolclass=poolclass,
        connect_args=connect_args,
        pool_pre_ping=True,
        **engine_args,
    )

    return engine


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create session factory.
    """

    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_database(engine: AsyncEngine) -> None:
    """
    Initialize database schema.
    """

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_database(engine: AsyncEngine) -> None:
    """
    Drop all database tables.
    """

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def health_check(engine: AsyncEngine) -> bool:
    """
    Check database connectivity.
    """

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


class DatabaseManager:
    """
    Database manager for lifecycle operations.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize database connection."""

        if self._connected:
            raise RuntimeError("Database already connected")

        self.engine = get_async_engine(self.settings)
        self.sessionmaker = get_sessionmaker(self.engine)

        self._connected = True

    async def disconnect(self) -> None:
        """Close database connection."""

        if self.engine:
            await self.engine.dispose()

        self.engine = None
        self.sessionmaker = None
        self._connected = False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session.
        """

        if not self._connected or not self.sessionmaker:
            raise RuntimeError("Database not connected")

        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def __aenter__(self) -> "DatabaseManager":
        """Async context manager entry."""

        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""

        await self.disconnect()


# Global manager
_db_manager: DatabaseManager | None = None


def get_db_manager(settings: Settings | None = None) -> DatabaseManager:
    """
    Get global database manager instance.
    """

    global _db_manager

    if _db_manager is None:
        if settings is None:
            raise RuntimeError("Database manager not initialized")

        _db_manager = DatabaseManager(settings)

    return _db_manager


async def reset_db_manager() -> None:
    """Reset global database manager."""

    global _db_manager

    if _db_manager:
        await _db_manager.disconnect()

    _db_manager = None
