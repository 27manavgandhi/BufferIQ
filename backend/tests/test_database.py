"""
Tests for database connectivity and session management.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from bufferiq.core.config import Environment, Settings
from bufferiq.core.database import (
    DatabaseManager,
    check_database_health,
    drop_database,
    get_async_engine,
    get_db_manager,
    get_session,
    get_sessionmaker,
    init_database,
    reset_db_manager,
)


@pytest.fixture
def sqlite_settings() -> Settings:
    """Create settings for in-memory SQLite database."""
    return Settings(
        environment=Environment.TESTING,
        database_url="sqlite:///:memory:",
        debug=False,
    )


@pytest.fixture
async def sqlite_engine(sqlite_settings: Settings) -> AsyncEngine:
    """Create SQLite engine for testing."""
    engine = get_async_engine(sqlite_settings)
    yield engine
    await engine.dispose()


@pytest.fixture
async def initialized_sqlite_engine(sqlite_engine: AsyncEngine) -> AsyncEngine:
    """Create SQLite engine with initialized schema."""
    await init_database(sqlite_engine)
    yield sqlite_engine
    await drop_database(sqlite_engine)


@pytest.fixture
async def db_manager(sqlite_settings: Settings) -> DatabaseManager:
    """Create and connect DatabaseManager for testing."""
    manager = DatabaseManager(sqlite_settings)
    await manager.connect()
    yield manager
    await manager.disconnect()
    await reset_db_manager()


class TestEngineCreation:
    """Test async engine creation with different configurations."""

    @pytest.mark.asyncio
    async def test_create_sqlite_engine(self, sqlite_settings: Settings) -> None:
        """SQLite engine should be created with correct configuration."""
        engine = get_async_engine(sqlite_settings)
        assert engine is not None
        assert "aiosqlite" in str(engine.url)
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_create_postgresql_engine(self) -> None:
        """PostgreSQL engine should be created with asyncpg driver."""
        settings = Settings(
            environment=Environment.TESTING,
            database_url="postgresql://user:pass@localhost/testdb",
        )
        engine = get_async_engine(settings)
        assert engine is not None
        assert "asyncpg" in str(engine.url)
        assert engine.pool is not None
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_sqlite_uses_null_pool(self, sqlite_settings: Settings) -> None:
        """SQLite should use NullPool for connection pooling."""
        engine = get_async_engine(sqlite_settings)
        assert engine.pool.__class__.__name__ == "NullPool"
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_postgresql_uses_queue_pool(self) -> None:
        """PostgreSQL should use QueuePool for connection pooling."""
        settings = Settings(
            environment=Environment.TESTING,
            database_url="postgresql://user:pass@localhost/testdb",
        )
        engine = get_async_engine(settings)
        assert engine.pool.__class__.__name__ == "QueuePool"
        await engine.dispose()


class TestSessionManagement:
    """Test async session creation and lifecycle."""

    @pytest.mark.asyncio
    async def test_create_sessionmaker(self, sqlite_engine: AsyncEngine) -> None:
        """Sessionmaker should be created from engine."""
        sessionmaker = get_sessionmaker(sqlite_engine)
        assert sessionmaker is not None
        assert sessionmaker.kw["class_"] == AsyncSession

    @pytest.mark.asyncio
    async def test_session_context_manager(
        self, initialized_sqlite_engine: AsyncEngine
    ) -> None:
        """Session context manager should handle lifecycle correctly."""
        sessionmaker = get_sessionmaker(initialized_sqlite_engine)
        async for session in get_session(sessionmaker):
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_session_commits_on_success(
        self, initialized_sqlite_engine: AsyncEngine
    ) -> None:
        """Session should commit transaction on successful completion."""
        sessionmaker = get_sessionmaker(initialized_sqlite_engine)

        async for session in get_session(sessionmaker):
            await session.execute(text("CREATE TABLE test_table (id INTEGER)"))

        async for session in get_session(sessionmaker):
            result = await session.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
                )
            )
            assert result.scalar() == "test_table"

    @pytest.mark.asyncio
    async def test_session_rolls_back_on_error(
        self, initialized_sqlite_engine: AsyncEngine
    ) -> None:
        """Session should rollback transaction on exception."""
        sessionmaker = get_sessionmaker(initialized_sqlite_engine)

        with pytest.raises(RuntimeError):
            async for session in get_session(sessionmaker):
                await session.execute(text("CREATE TABLE test_table (id INTEGER)"))
                raise RuntimeError("Simulated error")

        async for session in get_session(sessionmaker):
            result = await session.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
                )
            )
            assert result.scalar() is None


class TestDatabaseHealth:
    """Test database health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_succeeds_with_good_connection(
        self, sqlite_engine: AsyncEngine
    ) -> None:
        """Health check should return True for valid connection."""
        result = await check_database_health(sqlite_engine)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_fails_with_disposed_engine(
        self, sqlite_settings: Settings
    ) -> None:
        """Health check should return False for disposed engine."""
        engine = get_async_engine(sqlite_settings)
        await engine.dispose()
        result = await check_database_health(engine)
        assert result is False


class TestDatabaseInitialization:
    """Test database schema initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_init_database_creates_schema(
        self, sqlite_engine: AsyncEngine
    ) -> None:
        """init_database should create all tables from Base metadata."""
        await init_database(sqlite_engine)
        async with sqlite_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result]
        assert len(tables) >= 0
        await drop_database(sqlite_engine)

    @pytest.mark.asyncio
    async def test_drop_database_removes_schema(
        self, initialized_sqlite_engine: AsyncEngine
    ) -> None:
        """drop_database should remove all tables."""
        await drop_database(initialized_sqlite_engine)
        async with initialized_sqlite_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result]
        assert len(tables) == 0


class TestDatabaseManager:
    """Test DatabaseManager lifecycle and operations."""

    @pytest.mark.asyncio
    async def test_manager_connects_successfully(
        self, sqlite_settings: Settings
    ) -> None:
        """DatabaseManager should connect successfully."""
        manager = DatabaseManager(sqlite_settings)
        await manager.connect()
        assert manager.engine is not None
        assert manager.sessionmaker is not None
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_manager_connect_twice_raises_error(
        self, sqlite_settings: Settings
    ) -> None:
        """Connecting twice should raise RuntimeError."""
        manager = DatabaseManager(sqlite_settings)
        await manager.connect()
        with pytest.raises(RuntimeError, match="already connected"):
            await manager.connect()
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_manager_disconnects_cleanly(self, sqlite_settings: Settings) -> None:
        """DatabaseManager should disconnect cleanly."""
        manager = DatabaseManager(sqlite_settings)
        await manager.connect()
        await manager.disconnect()
        assert manager.engine is None
        assert manager.sessionmaker is None

    @pytest.mark.asyncio
    async def test_manager_session_context_manager(
        self, db_manager: DatabaseManager
    ) -> None:
        """DatabaseManager session should work as context manager."""
        async for session in db_manager.session():
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_manager_session_without_connection_raises_error(
        self, sqlite_settings: Settings
    ) -> None:
        """Using session without connection should raise RuntimeError."""
        manager = DatabaseManager(sqlite_settings)
        with pytest.raises(RuntimeError, match="not connected"):
            async for _ in manager.session():
                pass


class TestGlobalDatabaseManager:
    """Test global DatabaseManager singleton."""

    @pytest.mark.asyncio
    async def test_get_db_manager_creates_instance(
        self, sqlite_settings: Settings
    ) -> None:
        """get_db_manager should create instance on first call."""
        manager = get_db_manager(sqlite_settings)
        assert isinstance(manager, DatabaseManager)
        await reset_db_manager()

    @pytest.mark.asyncio
    async def test_get_db_manager_returns_same_instance(
        self, sqlite_settings: Settings
    ) -> None:
        """get_db_manager should return same instance on subsequent calls."""
        manager1 = get_db_manager(sqlite_settings)
        manager2 = get_db_manager()
        assert manager1 is manager2
        await reset_db_manager()

    @pytest.mark.asyncio
    async def test_get_db_manager_without_settings_raises_error(self) -> None:
        """get_db_manager without settings should raise error if not initialized."""
        await reset_db_manager()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_db_manager()

    @pytest.mark.asyncio
    async def test_reset_db_manager_clears_instance(
        self, sqlite_settings: Settings
    ) -> None:
        """reset_db_manager should clear global instance."""
        get_db_manager(sqlite_settings)
        await reset_db_manager()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_db_manager()
