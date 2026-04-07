"""
Response cache with Redis backend and TTL support.
"""

import json
from typing import Any, cast

from redis.asyncio import Redis


class ResponseCache:
    """
    Redis-backed response cache with TTL.

    Caches API responses to reduce API calls and improve performance.
    """

    def __init__(self, redis: Redis, default_ttl: int = 300) -> None:
        """
        Initialize cache.

        Args:
            redis: Redis client
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.redis = redis
        self.default_ttl = default_ttl

    def _get_key(self, query: str, variables: dict[str, Any] | None = None) -> str:
        """
        Generate cache key from query and variables.

        Args:
            query: GraphQL query
            variables: Query variables

        Returns:
            Cache key
        """
        var_str = json.dumps(variables or {}, sort_keys=True)
        # Simple hash of query + variables
        return f"cache:{hash(query + var_str)}"

    async def get(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Get cached response.

        Args:
            query: GraphQL query
            variables: Query variables

        Returns:
            Cached response or None
        """
        key = self._get_key(query, variables)
        data_bytes = await self.redis.get(key)

        if data_bytes is None:
            return None

        data_str = (
            data_bytes if isinstance(data_bytes, str) else data_bytes.decode("utf-8")
        )

        # 🔥 FIX: cast result of json.loads
        return cast(dict[str, Any], json.loads(data_str))

    async def set(
        self,
        query: str,
        variables: dict[str, Any] | None,
        response: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """
        Cache response.

        Args:
            query: GraphQL query
            variables: Query variables
            response: Response to cache
            ttl: TTL in seconds (uses default if None)
        """
        key = self._get_key(query, variables)
        data = json.dumps(response)
        await self.redis.setex(key, ttl or self.default_ttl, data)

    async def invalidate(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> None:
        """
        Invalidate cached response.

        Args:
            query: GraphQL query
            variables: Query variables
        """
        key = self._get_key(query, variables)
        await self.redis.delete(key)

    async def clear(self) -> None:
        """Clear all cached responses."""
        keys = []
        async for key in self.redis.scan_iter(match="cache:*"):
            keys.append(key)

        if keys:
            await self.redis.delete(*keys)
