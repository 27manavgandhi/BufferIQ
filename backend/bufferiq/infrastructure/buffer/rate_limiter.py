"""
Token bucket rate limiter with Redis backend.

Implements multi-tier rate limiting:
- 100 requests per 15 minutes
- 500 requests per 24 hours
- 10,000 requests per 30 days
"""

from typing import Literal

from redis.asyncio import Redis

from bufferiq.infrastructure.buffer.exceptions import BufferRateLimitError


class RateLimiter:
    """
    Token bucket rate limiter with Redis storage.

    Uses Redis for distributed rate limiting across multiple instances.
    """

    def __init__(
        self,
        redis: Redis,
        max_requests_15min: int = 100,
        max_requests_24hr: int = 500,
        max_requests_30day: int = 10000,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            redis: Redis client for storing tokens
            max_requests_15min: Maximum requests per 15 minutes
            max_requests_24hr: Maximum requests per 24 hours
            max_requests_30day: Maximum requests per 30 days
        """
        self.redis = redis
        self.limits = {
            "15min": (max_requests_15min, 15 * 60),
            "24hr": (max_requests_24hr, 24 * 60 * 60),
            "30day": (max_requests_30day, 30 * 24 * 60 * 60),
        }

    def _get_key(self, user_id: str, window: Literal["15min", "24hr", "30day"]) -> str:
        """
        Get Redis key for rate limit window.

        Args:
            user_id: User identifier
            window: Time window (15min, 24hr, 30day)

        Returns:
            Redis key
        """
        return f"rate_limit:{user_id}:{window}"

    async def check_limit(self, user_id: str, window: str | None = None) -> None:
        """
        Check if request is allowed under all rate limits.

        Args:
            user_id: User identifier

        Raises:
            BufferRateLimitError: If any rate limit is exceeded
        """
        windows = [window] if window else self.limits.keys()

        for window in windows:
            max_requests, window_seconds = self.limits[window]
            key = self._get_key(user_id, window)  # type: ignore
            count_bytes = await self.redis.get(key)

            if count_bytes is not None:
                count = int(count_bytes)
                if count >= max_requests:
                    ttl = await self.redis.ttl(key)
                    raise BufferRateLimitError(
                        f"Rate limit exceeded for {window} window",
                        retry_after=ttl if ttl > 0 else window_seconds,
                    )

    async def increment(self, user_id: str) -> None:
        """
        Increment request counters for all windows.

        Args:
            user_id: User identifier
        """
        for window, (_, window_seconds) in self.limits.items():
            key = self._get_key(user_id, window)  # type: ignore

            # Increment counter
            count = await self.redis.incr(key)

            # Set expiry on first request
            if count == 1:
                await self.redis.expire(key, window_seconds)

    async def reset(self, user_id: str) -> None:
        """
        Reset all rate limit counters for a user.

        Args:
            user_id: User identifier
        """
        for window in self.limits.keys():
            key = self._get_key(user_id, window)  # type: ignore
            await self.redis.delete(key)

    async def get_remaining(
        self, user_id: str, window: Literal["15min", "24hr", "30day"]
    ) -> int:
        """
        Get remaining requests for a time window.

        Args:
            user_id: User identifier
            window: Time window

        Returns:
            Number of remaining requests
        """
        max_requests, _ = self.limits[window]
        key = self._get_key(user_id, window)
        count_bytes = await self.redis.get(key)

        if count_bytes is None:
            return max_requests

        count = int(count_bytes)
        return max(0, max_requests - count)

    async def get_reset_time(
        self, user_id: str, window: Literal["15min", "24hr", "30day"]
    ) -> int:
        """
        Get time until rate limit resets (seconds).

        Args:
            user_id: User identifier
            window: Time window

        Returns:
            Seconds until reset
        """
        key = self._get_key(user_id, window)
        ttl = await self.redis.ttl(key)

        if ttl < 0:
            return 0

        return int(ttl)
