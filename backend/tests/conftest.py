"""
Pytest configuration and fixtures.

Provides reusable fixtures for testing database models, Redis, and Buffer API client.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone

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
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter
from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker
from bufferiq.infrastructure.sync.sync_service import SyncService
from bufferiq.infrastructure.sync.transformers import BufferTransformer

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
    """Create test settings."""
    return Settings(
        environment=Environment.TESTING,
        database_url="sqlite:///:memory:",
        buffer_api_url="https://graph.buffer.com/graphql",
        buffer_api_key="test_api_key",
        redis_url="redis://localhost:6379/1",
        debug=False,
    )


# ============================================================================
# Database Fixtures
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
# Domain Model Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def sample_user(test_session: AsyncSession) -> User:
    """Create sample user."""
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
    """Create sample organization."""
    org = Organization(
        user_id=sample_user.id,
        buffer_org_id="buffer_org_456",
        name="Test Organization",
    )
    test_session.add(org)
    await test_session.commit()
    await test_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def sample_channel(
    test_session: AsyncSession, sample_organization: Organization
) -> Channel:
    """Create sample channel."""
    channel = Channel(
        organization_id=sample_organization.id,
        buffer_channel_id="channel_789",
        platform="linkedin",
        handle="@testhandle",
        is_active=True,
    )
    test_session.add(channel)
    await test_session.commit()
    await test_session.refresh(channel)
    return channel


@pytest_asyncio.fixture
async def sample_post(test_session: AsyncSession, sample_channel: Channel) -> Post:
    """Create sample post."""
    post = Post(
        channel_id=sample_channel.id,
        buffer_post_id="post_101",
        content="Test post content about AI.",
        content_hash="abc123def456",
        status="sent",
        published_at=datetime.now(timezone.utc),
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
# Redis & External Services
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def redis_client() -> AsyncGenerator[Redis, None]:
    """Create Redis client for testing."""
    client = Redis.from_url("redis://localhost:6379/1", decode_responses=True)

    try:
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

    await client.flushdb()
    yield client

    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture(scope="function")
async def test_cache(redis_client: Redis) -> ResponseCache:
    """Create response cache."""
    cache = ResponseCache(redis_client, default_ttl=60)
    await cache.clear()
    return cache


@pytest_asyncio.fixture(scope="function")
async def test_rate_limiter(redis_client: Redis) -> RateLimiter:
    """Create rate limiter."""
    limiter = RateLimiter(
        redis_client,
        max_requests_15min=100,
        max_requests_24hr=500,
        max_requests_30day=10000,
    )
    await limiter.reset("test_user")
    return limiter


@pytest_asyncio.fixture(scope="function")
async def test_buffer_client(
    test_settings: Settings,
    test_rate_limiter: RateLimiter,
    test_cache: ResponseCache,
) -> AsyncGenerator[BufferClient, None]:
    """Create Buffer API client."""
    client = BufferClient(
        test_settings,
        test_rate_limiter,
        test_cache,
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0,
    )
    client.set_user_id("test_user")

    yield client

    await client.close()


# ============================================================================
# Sync Service Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def sync_transformer() -> BufferTransformer:
    return BufferTransformer()


@pytest_asyncio.fixture(scope="function")
async def sync_tracker(test_session: AsyncSession) -> ProgressTracker:
    return ProgressTracker(test_session)


@pytest_asyncio.fixture(scope="function")
async def sync_service(
    test_session: AsyncSession,
    test_buffer_client: BufferClient,
    sync_transformer: BufferTransformer,
    sync_tracker: ProgressTracker,
) -> SyncService:
    return SyncService(
        test_session,
        test_buffer_client,
        sync_transformer,
        sync_tracker,
    )


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires external services)",
    )
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests."""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        if (
            "redis_client" in item.fixturenames
            or "test_buffer_client" in item.fixturenames
        ):
            item.add_marker(pytest.mark.integration)