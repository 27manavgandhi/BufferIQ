"""
Database engine and session management using SQLAlchemy 2.0.

This module provides async database connectivity with:
- Connection pooling with configurable limits
- Health check functionality
- Graceful connection handling
- Transaction management
- Base model with common fields
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.sql import func

from bufferiq.core.config import Settings
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """
    Base class for all database models.

    Provides common fields and functionality for all models:
    - id: Primary key (auto-incrementing integer)
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    """

    pass


def get_async_engine(settings: Settings) -> AsyncEngine:
    """
    Create and configure async SQLAlchemy engine.

    Args:
        settings: Application settings containing database configuration

    Returns:
        Configured AsyncEngine instance

    Raises:
        ValueError: If database URL is invalid or unsupported
    """
    db_url = settings.database_url

    if settings.database_is_sqlite:
        async_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        poolclass = NullPool
        connect_args = {"check_same_thread": False}
    elif settings.database_is_postgresql:
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        poolclass = QueuePool
        connect_args = {
            "server_settings": {
                "application_name": "bufferiq",
                "jit": "off",
            }
        }
    else:
        raise ValueError(
            f"Unsupported database URL: {db_url}. "
            "Use 'sqlite:///' or 'postgresql://'"
        )

    engine = create_async_engine(
        async_url,
        echo=settings.get_database_echo(),
        poolclass=poolclass,
        pool_size=5 if poolclass == QueuePool else None,
        max_overflow=10 if poolclass == QueuePool else None,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args=connect_args,
    )

    if settings.is_development:
        _setup_query_logging(engine)

    logger.info(
        f"Database engine created: {async_url.split('@')[-1] if '@' in async_url else async_url}"
    )

    return engine


def _setup_query_logging(engine: AsyncEngine) -> None:
    """
    Set up query logging for development environment.

    Args:
        engine: SQLAlchemy engine to attach logging to
    """

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def receive_before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        logger.debug(f"SQL Query: {statement}")
        logger.debug(f"Parameters: {parameters}")


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create configured async session factory.

    Args:
        engine: AsyncEngine to bind sessions to

    Returns:
        async_sessionmaker configured for the engine
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@asynccontextmanager
async def get_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions with automatic cleanup.

    Usage:
        async with get_session(sessionmaker) as session:
            result = await session.execute(select(User))
            users = result.scalars().all()

    Args:
        sessionmaker: async_sessionmaker to create session from

    Yields:
        AsyncSession instance

    Raises:
        Exception: Re-raises any exception after rollback
    """
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health(engine: AsyncEngine) -> bool:
    """
    Verify database connection is healthy.

    Args:
        engine: AsyncEngine to check

    Returns:
        True if database is accessible and responding, False otherwise
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database health check: OK")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def init_database(engine: AsyncEngine) -> None:
    """
    Initialize database schema.

    Creates all tables defined in Base metadata.
    Should only be used in development/testing.
    In production, use Alembic migrations.

    Args:
        engine: AsyncEngine to create tables with

    Raises:
        Exception: If table creation fails
    """
    logger.info("Initializing database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully")


async def drop_database(engine: AsyncEngine) -> None:
    """
    Drop all database tables.

    WARNING: This deletes all data. Only use in development/testing.

    Args:
        engine: AsyncEngine to drop tables from

    Raises:
        Exception: If table drop fails
    """
    logger.warning("Dropping all database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All database tables dropped")


class DatabaseManager:
    """
    Centralized database management.

    Handles engine and sessionmaker lifecycle for the application.
    """

    def __init__(self, settings: Settings):
        """
        Initialize database manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.engine: Optional[AsyncEngine] = None
        self.sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None

    async def connect(self) -> None:
        """
        Establish database connection.

        Creates engine and sessionmaker.

        Raises:
            RuntimeError: If already connected
        """
        if self.engine is not None:
            raise RuntimeError("Database already connected")

        self.engine = get_async_engine(self.settings)
        self.sessionmaker = get_sessionmaker(self.engine)

        health_ok = await check_database_health(self.engine)
        if not health_ok:
            raise RuntimeError("Database health check failed after connection")

        logger.info("Database manager connected")

    async def disconnect(self) -> None:
        """
        Close database connection.

        Disposes engine and clears sessionmaker.
        """
        if self.engine is None:
            return

        await self.engine.dispose()
        self.engine = None
        self.sessionmaker = None

        logger.info("Database manager disconnected")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session context manager.

        Usage:
            async with db_manager.session() as session:
                result = await session.execute(select(User))

        Yields:
            AsyncSession instance

        Raises:
            RuntimeError: If not connected
        """
        if self.sessionmaker is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with get_session(self.sessionmaker) as session:
            yield session


# Global instance for application use
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(settings: Optional[Settings] = None) -> DatabaseManager:
    """
    Get or create global DatabaseManager instance.

    Args:
        settings: Application settings (required on first call)

    Returns:
        DatabaseManager instance

    Raises:
        RuntimeError: If called without settings before initialization
    """
    global _db_manager

    if _db_manager is None:
        if settings is None:
            raise RuntimeError(
                "DatabaseManager not initialized. "
                "Call with settings parameter first."
            )
        _db_manager = DatabaseManager(settings)

    return _db_manager


async def reset_db_manager() -> None:
    """
    Reset global DatabaseManager.

    Useful for testing to ensure clean state between tests.
    """
    global _db_manager

    if _db_manager is not None:
        await _db_manager.disconnect()
        _db_manager = None