"""
Pytest configuration and fixtures.

Provides reusable fixtures for testing database models.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bufferiq.core.config import Environment, Settings
from bufferiq.core.database import (
    Base,
    drop_database,
    get_async_engine,
    get_sessionmaker,
    init_database,
)
from bufferiq.domain.models import Channel, Organization, Post, User


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with in-memory SQLite."""
    return Settings(
        environment=Environment.TESTING,
        database_url="sqlite+aiosqlite:///:memory:",
        debug=False,
    )


@pytest_asyncio.fixture(scope="function")
async def test_engine(test_settings: Settings) -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine."""
    engine = get_async_engine(test_settings)
    await init_database(engine)
    yield engine
    await drop_database(engine)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_sessionmaker(
    test_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create test session factory."""
    return get_sessionmaker(test_engine)


@pytest_asyncio.fixture(scope="function")
async def test_session(
    test_sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async with test_sessionmaker() as session:
        yield session


@pytest_asyncio.fixture
async def sample_user(test_session: AsyncSession) -> User:
    """Create sample user for testing."""
    user = User(
        buffer_org_id="org_123",
        buffer_access_token="token_abc123",
        email="test@example.com",
        is_active=True,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def sample_organization(
    test_session: AsyncSession, sample_user: User
) -> Organization:
    """Create sample organization for testing."""
    org = Organization(
        user_id=sample_user.id, buffer_org_id="buffer_org_456", name="Test Organization"
    )
    test_session.add(org)
    await test_session.commit()
    await test_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def sample_channel(
    test_session: AsyncSession, sample_organization: Organization
) -> Channel:
    """Create sample channel for testing."""
    channel = Channel(
        organization_id=sample_organization.id,
        buffer_channel_id="channel_789",
        platform="linkedin",
        handle="testhandle",
        is_active=True,
    )
    test_session.add(channel)
    await test_session.commit()
    await test_session.refresh(channel)
    return channel


@pytest_asyncio.fixture
async def sample_post(test_session: AsyncSession, sample_channel: Channel) -> Post:
    """Create sample post for testing."""
    post = Post(
        channel_id=sample_channel.id,
        buffer_post_id="post_101",
        content="This is a test post about AI and technology.",
        content_hash="abc123def456",
        status="sent",
        likes=50,
        comments=10,
        shares=5,
        impressions=1000,
        engagement_rate=0.065,
    )
    test_session.add(post)
    await test_session.commit()
    await test_session.refresh(post)
    return post