from typing import Any, AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool, Pool

from bufferiq.core.config import Settings
from bufferiq.domain.base import Base


# ---------------- ENGINE ---------------- #


def get_async_engine(settings: Settings) -> AsyncEngine:
    database_url = settings.database_url

    if database_url.startswith("sqlite://"):
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        poolclass: type[Pool] = NullPool
        connect_args = {"check_same_thread": False}

    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        poolclass = QueuePool
        connect_args = {}

    else:
        raise ValueError("Unsupported database URL")

    return create_async_engine(
        database_url,
        echo=getattr(settings, "database_echo", False),
        poolclass=poolclass,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_size=5 if poolclass == QueuePool else 0,
        max_overflow=10 if poolclass == QueuePool else 0,
    )


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


# ---------------- SESSION CONTEXT ---------------- #


async def get_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------- DB INIT ---------------- #


async def init_database(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_database(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ---------------- HEALTH ---------------- #


async def health_check(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def check_database_health(engine: AsyncEngine) -> bool:
    return await health_check(engine)


# ---------------- DATABASE MANAGER ---------------- #


class DatabaseManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Optional[AsyncEngine] = None
        self.sessionmaker: Optional[async_sessionmaker] = None
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            raise RuntimeError("Database already connected")

        self.engine = get_async_engine(self.settings)
        self.sessionmaker = get_sessionmaker(self.engine)
        self._connected = True

    async def disconnect(self) -> None:
        if self.engine:
            await self.engine.dispose()

        self.engine = None
        self.sessionmaker = None
        self._connected = False

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if not self._connected or not self.sessionmaker:
            raise RuntimeError("Database not connected")

        async with get_session(self.sessionmaker) as session:
            yield session


# ---------------- GLOBAL MANAGER ---------------- #

_db_manager: Optional[DatabaseManager] = None


def get_db_manager(settings: Optional[Settings] = None) -> DatabaseManager:
    global _db_manager

    if _db_manager is None:
        if settings is None:
            raise RuntimeError("Database manager not initialized")
        _db_manager = DatabaseManager(settings)

    return _db_manager


async def reset_db_manager() -> None:
    global _db_manager

    if _db_manager:
        await _db_manager.disconnect()

    _db_manager = None
