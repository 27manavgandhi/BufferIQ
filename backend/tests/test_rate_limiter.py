"""
Tests for rate limiter.
"""

import pytest
from redis.asyncio import Redis

from bufferiq.infrastructure.buffer.exceptions import BufferRateLimitError
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter


@pytest.fixture
async def redis_client() -> Redis:
    """Create Redis client for testing."""
    return Redis.from_url("redis://localhost:6379", decode_responses=True)


@pytest.fixture
async def rate_limiter(redis_client: Redis) -> RateLimiter:
    """Create rate limiter for testing."""
    limiter = RateLimiter(
        redis_client, max_requests_15min=5, max_requests_24hr=10, max_requests_30day=20
    )
    # Clear test user data
    await limiter.reset("test_user")
    return limiter


@pytest.mark.asyncio
class TestRateLimiter:
    """Test rate limiter functionality."""

    async def test_initial_state(self, rate_limiter: RateLimiter) -> None:
        """Initial state should have full quota."""
        remaining_15min = await rate_limiter.get_remaining("test_user", "15min")
        remaining_24hr = await rate_limiter.get_remaining("test_user", "24hr")
        remaining_30day = await rate_limiter.get_remaining("test_user", "30day")

        assert remaining_15min == 5
        assert remaining_24hr == 10
        assert remaining_30day == 20

    async def test_check_limit_allows_request(self, rate_limiter: RateLimiter) -> None:
        """Should allow request when under limit."""
        # Should not raise
        await rate_limiter.check_limit("test_user")

    async def test_increment_reduces_quota(self, rate_limiter: RateLimiter) -> None:
        """Increment should reduce remaining quota."""
        await rate_limiter.increment("test_user")

        remaining_15min = await rate_limiter.get_remaining("test_user", "15min")
        remaining_24hr = await rate_limiter.get_remaining("test_user", "24hr")

        assert remaining_15min == 4
        assert remaining_24hr == 9

    async def test_rate_limit_exceeded(self, rate_limiter: RateLimiter) -> None:
        """Should raise error when limit exceeded."""
        # Use up all requests for 15min window
        for _ in range(5):
            await rate_limiter.increment("test_user")

        # Next check should raise
        with pytest.raises(BufferRateLimitError) as exc_info:
            await rate_limiter.check_limit("test_user")

        assert exc_info.value.status_code == 429
        assert "15min" in str(exc_info.value)
        assert exc_info.value.retry_after is not None

    async def test_reset_clears_counters(self, rate_limiter: RateLimiter) -> None:
        """Reset should clear all counters."""
        # Make some requests
        for _ in range(3):
            await rate_limiter.increment("test_user")

        # Reset
        await rate_limiter.reset("test_user")

        # Should have full quota again
        remaining = await rate_limiter.get_remaining("test_user", "15min")
        assert remaining == 5

    async def test_get_reset_time(self, rate_limiter: RateLimiter) -> None:
        """Should return time until reset."""
        await rate_limiter.increment("test_user")

        reset_time = await rate_limiter.get_reset_time("test_user", "15min")
        assert reset_time > 0
        assert reset_time <= 15 * 60  # Within 15 minutes

    async def test_different_users_independent(self, rate_limiter: RateLimiter) -> None:
        """Different users should have independent quotas."""
        # User 1 uses quota
        for _ in range(5):
            await rate_limiter.increment("user_1")

        # User 1 should be rate limited
        with pytest.raises(BufferRateLimitError):
            await rate_limiter.check_limit("user_1")

        # User 2 should still have quota
        remaining = await rate_limiter.get_remaining("user_2", "15min")
        assert remaining == 5

        # Clean up
        await rate_limiter.reset("user_1")
        await rate_limiter.reset("user_2")

    async def test_multiple_window_limits(self, rate_limiter: RateLimiter) -> None:
        """Should enforce limits across multiple windows."""
        # Use up 15min quota
        for _ in range(5):
            await rate_limiter.increment("test_user")

        # Should fail 15min check
        with pytest.raises(BufferRateLimitError) as exc_info:
            await rate_limiter.check_limit("test_user")
        assert "15min" in str(exc_info.value)

        # Reset 15min window
        await rate_limiter.reset("test_user")

        # Use up 24hr quota
        for _ in range(10):
            await rate_limiter.increment("test_user")

        # Should fail 24hr check
        with pytest.raises(BufferRateLimitError) as exc_info:
            await rate_limiter.check_limit("test_user")
        assert "24hr" in str(exc_info.value)
