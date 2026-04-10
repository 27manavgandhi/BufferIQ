"""
Alembic environment configuration.

Handles database connection and migration context setup for both
offline (SQL generation) and online (direct execution) modes.
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from bufferiq.core.config import Settings
from bufferiq.core.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = Settings()


def get_url() -> str:
    """
    Get database URL from environment or settings.
    """

    # First priority: environment variable
    db_url = os.getenv("DATABASE_URL", settings.database_url)

    # SQLite
    if "sqlite" in db_url:
        return db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

    # PostgreSQL
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+asyncpg://")

    if db_url.startswith("postgresql+asyncpg://"):
        return db_url

    raise ValueError(f"Unsupported database URL: {db_url}")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    """

    url = get_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Execute migrations using provided connection.
    """

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using async engine."""

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in online mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
