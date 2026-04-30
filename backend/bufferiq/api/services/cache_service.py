"""Redis-based response caching service."""

import hashlib
import json
from typing import Optional

import aioredis
from pydantic import BaseModel

from bufferiq.api.models.prediction import PredictionRequest, PredictionResponse
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Redis-based response caching.

    Caches prediction responses to reduce computation.
    Uses TTL to ensure freshness.
    """

    def __init__(self, redis_url: str = "redis://localhost", ttl: int = 3600):
        """
        Initialize cache service.

        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if not self._connected:
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                await self.redis.ping()
                self._connected = True
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Don't raise - gracefully degrade without cache
                self.redis = None

    def generate_key(
        self, request: PredictionRequest, model_name: str = "default"
    ) -> str:
        """
        Generate cache key from request.

        Args:
            request: Prediction request
            model_name: Model identifier

        Returns:
            Cache key (MD5 hash)
        """
        # Create deterministic key from request
        key_parts = {
            "content": request.content,
            "platform": request.platform,
            "scheduled_time": (
                request.scheduled_time.isoformat()
                if request.scheduled_time
                else None
            ),
            "has_media": request.has_media,
            "has_link": request.has_link,
            "model": model_name,
        }

        key_json = json.dumps(key_parts, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()

        return f"pred:{key_hash}"

    async def get(self, key: str) -> Optional[PredictionResponse]:
        """
        Get cached response.

        Args:
            key: Cache key

        Returns:
            Cached response or None if not found
        """
        await self._ensure_connected()

        if not self.redis:
            return None

        try:
            data = await self.redis.get(key)
            if data:
                logger.debug(f"Cache hit: {key}")
                return PredictionResponse.parse_raw(data)

            logger.debug(f"Cache miss: {key}")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, response: PredictionResponse) -> None:
        """
        Cache response with TTL.

        Args:
            key: Cache key
            response: Response to cache
        """
        await self._ensure_connected()

        if not self.redis:
            return

        try:
            await self.redis.setex(key, self.ttl, response.json())
            logger.debug(f"Cached response: {key}")

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def invalidate(self, pattern: str = "pred:*") -> int:
        """
        Invalidate cache keys matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            Number of keys deleted
        """
        await self._ensure_connected()

        if not self.redis:
            return 0

        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Invalidated {deleted} cache keys")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0

    async def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        await self._ensure_connected()

        if not self.redis:
            return {"status": "disconnected"}

        try:
            info = await self.redis.info()
            dbsize = await self.redis.dbsize()

            return {
                "status": "connected",
                "keys": dbsize,
                "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0)
                        + info.get("keyspace_misses", 0),
                        1,
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"status": "error", "message": str(e)}

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Redis connection closed")