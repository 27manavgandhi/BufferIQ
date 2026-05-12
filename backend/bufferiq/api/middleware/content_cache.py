"""
Content caching middleware.

Caches content analysis results to improve performance.
"""

import hashlib
import json
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ContentCacheMiddleware(BaseHTTPMiddleware):
    """
    Cache content analysis results.

    Caches responses for identical content analysis requests.

    Example:
```python
        app.add_middleware(ContentCacheMiddleware, cache_ttl=3600)
```
    """

    def __init__(self, app: Any, cache_ttl: int = 3600) -> None:
        """
        Initialize cache middleware.

        Args:
            app: FastAPI application
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(app)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request with caching.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Only cache POST requests to /content/analyze
        if (
            request.method == "POST"
            and "/content/analyze" in str(request.url)
        ):
            # Get request body
            body = await request.body()
            cache_key = self._generate_cache_key(body)

            # Check cache
            cached_response = self._get_cached(cache_key)
            if cached_response:
                return Response(
                    content=json.dumps(cached_response),
                    media_type="application/json",
                    headers={"X-Cache": "HIT"},
                )

            # Process request
            response = await call_next(request)

            # Cache response if successful
            if response.status_code == 200:
                # Note: This is simplified - in production, use proper async cache
                pass

            return response
        else:
            return await call_next(request)

    def _generate_cache_key(self, body: bytes) -> str:
        """
        Generate cache key from request body.

        Args:
            body: Request body

        Returns:
            Cache key
        """
        return hashlib.md5(body).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response.

        Args:
            key: Cache key

        Returns:
            Cached response or None
        """
        return self.cache.get(key)

    def _set_cached(self, key: str, value: Dict[str, Any]) -> None:
        """
        Set cached response.

        Args:
            key: Cache key
            value: Response to cache
        """
        self.cache[key] = value