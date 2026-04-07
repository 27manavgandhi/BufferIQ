"""
Tests for response cache.
"""

import pytest
from redis.asyncio import Redis

from bufferiq.infrastructure.buffer.cache import ResponseCache


@pytest.fixture
async def redis_client() -> Redis:
    """Create Redis client for testing."""
    return Redis.from_url("redis://localhost:6379", decode_responses=True)


@pytest.fixture
async def cache(redis_client: Redis) -> ResponseCache:
    """Create cache for testing."""
    cache = ResponseCache(redis_client, default_ttl=60)
    await cache.clear()
    return cache


@pytest.mark.asyncio
class TestResponseCache:
    """Test response cache functionality."""

    async def test_get_empty_cache(self, cache: ResponseCache) -> None:
        """Should return None for cache miss."""
        result = await cache.get("query { test }", {})
        assert result is None

    async def test_set_and_get(self, cache: ResponseCache) -> None:
        """Should store and retrieve cached response."""
        query = "query { organizations { id } }"
        variables = {"limit": 10}
        response = {"data": {"organizations": [{"id": "1"}]}}

        await cache.set(query, variables, response)
        cached = await cache.get(query, variables)

        assert cached == response

    async def test_different_variables_different_cache(
        self, cache: ResponseCache
    ) -> None:
        """Different variables should use different cache keys."""
        query = "query { organizations { id } }"

        response_1 = {"data": {"organizations": [{"id": "1"}]}}
        response_2 = {"data": {"organizations": [{"id": "2"}]}}

        await cache.set(query, {"limit": 10}, response_1)
        await cache.set(query, {"limit": 20}, response_2)

        cached_1 = await cache.get(query, {"limit": 10})
        cached_2 = await cache.get(query, {"limit": 20})

        assert cached_1 == response_1
        assert cached_2 == response_2

    async def test_invalidate(self, cache: ResponseCache) -> None:
        """Should invalidate cached response."""
        query = "query { test }"
        response = {"data": {"test": "value"}}

        await cache.set(query, None, response)
        await cache.invalidate(query, None)

        cached = await cache.get(query, None)
        assert cached is None

    async def test_clear_all(self, cache: ResponseCache) -> None:
        """Should clear all cached responses."""
        await cache.set("query1", None, {"data": "1"})
        await cache.set("query2", None, {"data": "2"})

        await cache.clear()

        cached_1 = await cache.get("query1", None)
        cached_2 = await cache.get("query2", None)

        assert cached_1 is None
        assert cached_2 is None

    async def test_custom_ttl(self, cache: ResponseCache) -> None:
        """Should respect custom TTL."""
        query = "query { test }"
        response = {"data": {"test": "value"}}

        # Set with 1 second TTL
        await cache.set(query, None, response, ttl=1)

        # Should be cached immediately
        cached = await cache.get(query, None)
        assert cached == response

        # Wait for expiry
        import asyncio

        await asyncio.sleep(1.1)

        # Should be expired
        cached = await cache.get(query, None)
        assert cached is None

    async def test_none_variables_consistent(self, cache: ResponseCache) -> None:
        """None variables and {} should use same cache key."""
        query = "query { test }"
        response = {"data": {"test": "value"}}

        await cache.set(query, None, response)

        # Should get same result with {} or None
        cached_none = await cache.get(query, None)
        cached_empty = await cache.get(query, {})

        assert cached_none == response
        assert cached_empty == response
