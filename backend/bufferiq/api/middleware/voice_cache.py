"""
Voice response caching middleware.

Caches voice analysis responses for improved performance.
"""

from typing import Optional, Dict, Any
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class VoiceCacheMiddleware:
    """
    Cache middleware for voice responses.
    
    Caches analysis results to reduce computation
    for repeated requests.
    """
    
    def __init__(self, cache_client: Optional[Any] = None, ttl: int = 3600):
        """
        Initialize cache middleware.
        
        Args:
            cache_client: Cache client (e.g., Redis)
            ttl: Cache TTL in seconds
        """
        self.cache = cache_client
        self.ttl = ttl
        self.enabled = cache_client is not None
    
    def generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """
        Generate cache key from request data.
        
        Args:
            request_data: Request parameters
        
        Returns:
            Cache key
        """
        # Create deterministic hash from request
        serialized = json.dumps(request_data, sort_keys=True)
        hash_obj = hashlib.md5(serialized.encode())
        return f"voice_analysis:{hash_obj.hexdigest()}"
    
    async def get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached response or None
        """
        if not self.enabled:
            return None
        
        try:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set_cached(
        self, cache_key: str, response_data: Dict[str, Any]
    ) -> None:
        """
        Cache response.
        
        Args:
            cache_key: Cache key
            response_data: Response to cache
        """
        if not self.enabled:
            return
        
        try:
            serialized = json.dumps(response_data)
            await self.cache.setex(cache_key, self.ttl, serialized)
            logger.info(f"Cached response: {cache_key}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")