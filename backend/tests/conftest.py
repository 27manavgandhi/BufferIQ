"""
Pytest configuration and fixtures.

Provides reusable fixtures for testing database models and Buffer API client.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bufferiq.core.cache import ResponseCache
from bufferiq.core.config import Environment, Settings
from bufferiq.core.database import (
    drop_database,
    get_async_engine,
    get_sessionmaker,
    init_database,
)
from bufferiq.domain.models import Channel, Organization, Post, User
from bufferiq.infrastructure.buffer_client import BufferClient
from bufferiq.infrastructure.rate_limiter import RateLimiter


# ============================================================================
# Event Loop & Settings
# ============================================================================


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
        database_url="sqlite:///:memory:",
        buffer_api_url="https://api.buffer.com/graphql",
        buffer_api_key="test_api_key_for_testing",
        debug=False,
    )


# ============================================================================
# Database Fixtures (Days 1-3)
# ============================================================================


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


# ============================================================================
# Domain Model Fixtures (Day 3)
# ============================================================================


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


# ============================================================================
# Redis & Buffer Client Fixtures (Day 4)
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def redis_client() -> AsyncGenerator[Redis, None]:
    """
    Create Redis client for testing.
    
    Connects to localhost Redis (must be running via docker-compose).
    """
    client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    
    # Test connection
    try:
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    
    yield client
    
    # Cleanup: flush test data
    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture(scope="function")
async def test_cache(redis_client: Redis) -> ResponseCache:
    """Create response cache for testing."""
    cache = ResponseCache(redis_client, default_ttl=60)
    await cache.clear()
    return cache


@pytest_asyncio.fixture(scope="function")
async def test_rate_limiter(redis_client: Redis) -> RateLimiter:
    """Create rate limiter for testing."""
    limiter = RateLimiter(
        redis_client,
        max_requests_15min=100,
        max_requests_24hr=500,
        max_requests_30day=10000,
    )
    # Clear any existing test user data
    await limiter.reset("test_user")
    return limiter


@pytest_asyncio.fixture(scope="function")
async def test_buffer_client(
    test_settings: Settings,
    test_rate_limiter: RateLimiter,
    test_cache: ResponseCache,
) -> AsyncGenerator[BufferClient, None]:
    """Create Buffer API client for testing."""
    client = BufferClient(
        test_settings,
        test_rate_limiter,
        test_cache,
        max_retries=3,
        base_delay=0.1,  # Faster retries for tests
        max_delay=1.0,   # Lower max delay for tests
    )
    client.set_user_id("test_user")
    yield client
    await client.close()


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark all async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Mark integration tests (tests that use Redis or external APIs)
        if "redis_client" in item.fixturenames or "test_buffer_client" in item.fixturenames:
            item.add_marker(pytest.mark.integration)