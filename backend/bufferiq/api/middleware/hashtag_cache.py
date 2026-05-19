"""
Hashtag caching middleware.

Implements caching for hashtag endpoints.
"""

from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import json


class HashtagCacheMiddleware(BaseHTTPMiddleware):
    """
    Cache middleware for hashtag endpoints.

    Caches responses for GET and POST requests to reduce load.
    """

    def __init__(
        self,
        app,
        cache_backend: Optional[any] = None,
        default_ttl: int = 3600,
    ) -> None:
        """
        Initialize cache middleware.

        Args:
            app: FastAPI application
            cache_backend: Cache backend (e.g., Redis)
            default_ttl: Default cache TTL in seconds
        """
        super().__init__(app)
        self.cache = cache_backend
        self.default_ttl = default_ttl

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request with caching.

        Args:
            request: Incoming request
            call_next: Next middleware

        Returns:
            Response (cached or fresh)
        """
        # Only cache hashtag endpoints
        if not request.url.path.startswith("/api/v1/hashtags"):
            return await call_next(request)

        # Only cache safe methods and specific POST endpoints
        cacheable_paths = [
            "/api/v1/hashtags/analyze",
            "/api/v1/hashtags/trends",
            "/api/v1/hashtags/insights",
        ]

        is_cacheable = (
            request.method == "GET"
            or (request.method == "POST" and request.url.path in cacheable_paths)
        )

        if not is_cacheable or not self.cache:
            return await call_next(request)

        # Generate cache key
        cache_key = await self._generate_cache_key(request)

        # Try to get from cache
        cached_response = await self._get_from_cache(cache_key)
        if cached_response:
            return Response(
                content=cached_response,
                media_type="application/json",
                headers={"X-Cache": "HIT"},
            )

        # Process request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Cache it
            await self._set_in_cache(cache_key, body)

            # Return response
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers) | {"X-Cache": "MISS"},
                media_type=response.media_type,
            )

        return response

    async def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        # Include path, query params, and body
        key_parts = [request.url.path]

        # Add query params
        if request.query_params:
            key_parts.append(str(request.query_params))

        # Add body for POST
        if request.method == "POST":
            body = await request.body()
            key_parts.append(body.decode())

        # Hash it
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return f"hashtag:cache:{key_hash}"

    async def _get_from_cache(self, key: str) -> Optional[bytes]:
        """Get value from cache."""
        if not self.cache:
            return None

        try:
            # Assuming Redis-like interface
            value = self.cache.get(key)
            return value
        except Exception:
            return None

    async def _set_in_cache(self, key: str, value: bytes) -> None:
        """Set value in cache."""
        if not self.cache:
            return

        try:
            # Assuming Redis-like interface
            self.cache.setex(key, self.default_ttl, value)
        except Exception:
            pass