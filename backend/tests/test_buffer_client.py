"""
Tests for Buffer API client.
"""

import pytest
from aiohttp import ClientError
from redis.asyncio import Redis

from bufferiq.core.config import Settings
from bufferiq.infrastructure.buffer.cache import ResponseCache
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.buffer.exceptions import (
    BufferAPIError,
    BufferAuthenticationError,
    BufferNetworkError,
    BufferRateLimitError,
    BufferValidationError,
)
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter


@pytest.fixture
async def redis_client() -> Redis:
    """Create Redis client for testing."""
    return Redis.from_url("redis://localhost:6379", decode_responses=True)


@pytest.fixture
async def rate_limiter(redis_client: Redis) -> RateLimiter:
    """Create rate limiter for testing."""
    limiter = RateLimiter(redis_client, max_requests_15min=10)
    # Clear any existing data
    await limiter.reset("test_user")
    return limiter


@pytest.fixture
async def cache(redis_client: Redis) -> ResponseCache:
    """Create cache for testing."""
    cache = ResponseCache(redis_client, default_ttl=60)
    await cache.clear()
    return cache


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        buffer_api_url="https://api.buffer.com/graphql",
        buffer_api_key="test_key",
    )


@pytest.fixture
async def client(
    settings: Settings, rate_limiter: RateLimiter, cache: ResponseCache
) -> BufferClient:
    """Create Buffer client for testing."""
    client = BufferClient(settings, rate_limiter, cache)
    client.set_user_id("test_user")
    yield client
    await client.close()


class TestBufferClient:
    """Test Buffer API client."""

    def test_client_initialization(
        self, settings: Settings, rate_limiter: RateLimiter, cache: ResponseCache
    ) -> None:
        """Client should initialize with correct settings."""
        client = BufferClient(settings, rate_limiter, cache)

        assert client.api_url == settings.buffer_api_url
        assert client.api_key == settings.buffer_api_key
        assert client.max_retries == 3
        assert client.base_delay == 1.0
        assert client.max_delay == 32.0

    def test_set_user_id(self, client: BufferClient) -> None:
        """Should set user ID for rate limiting."""
        client.set_user_id("user_123")
        assert client.user_id == "user_123"

    def test_get_headers(self, client: BufferClient) -> None:
        """Should return correct HTTP headers."""
        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test_key"
        assert "User-Agent" in headers

    def test_calculate_delay(self, client: BufferClient) -> None:
        """Should calculate exponential backoff with jitter."""
        # First attempt
        delay_0 = client._calculate_delay(0)
        assert 0.75 <= delay_0 <= 1.25  # 1.0 ± 25%

        # Second attempt
        delay_1 = client._calculate_delay(1)
        assert 1.5 <= delay_1 <= 2.5  # 2.0 ± 25%

        # Third attempt
        delay_2 = client._calculate_delay(2)
        assert 3.0 <= delay_2 <= 5.0  # 4.0 ± 25%

        # Should respect max_delay
        delay_10 = client._calculate_delay(10)
        assert delay_10 <= client.max_delay * 1.25  # Max + jitter


class TestBufferClientErrors:
    """Test error handling."""

    def test_authentication_error(self) -> None:
        """Should create authentication error."""
        error = BufferAuthenticationError()
        assert error.status_code == 401
        assert "Authentication failed" in str(error)

    def test_rate_limit_error(self) -> None:
        """Should create rate limit error with retry_after."""
        error = BufferRateLimitError(retry_after=60)
        assert error.status_code == 429
        assert error.retry_after == 60

    def test_validation_error(self) -> None:
        """Should create validation error with field errors."""
        errors = {"content": ["Content is required"]}
        error = BufferValidationError("Validation failed", errors=errors)
        assert error.status_code == 400
        assert error.errors == errors

    def test_network_error(self) -> None:
        """Should create network error with original exception."""
        original = ClientError("Connection failed")
        error = BufferNetworkError("Network error", original_error=original)
        assert error.status_code is None
        assert error.original_error == original


@pytest.mark.asyncio
class TestBufferClientIntegration:
    """Integration tests (require mocking or real API)."""

    async def test_health_check_success(self, client: BufferClient, mocker) -> None:
        """Health check should return True on success."""
        # Mock the query method to return success
        mocker.patch.object(client, "query", return_value={"__typename": "Query"})

        result = await client.health_check()
        assert result is True

    async def test_health_check_failure(self, client: BufferClient, mocker) -> None:
        """Health check should return False on failure."""
        # Mock the query method to raise exception
        mocker.patch.object(client, "query", side_effect=BufferAPIError("Error"))

        result = await client.health_check()
        assert result is False
