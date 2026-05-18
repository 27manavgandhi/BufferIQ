"""Caching middleware for gap analysis."""

import json
import hashlib
from typing import Any, Callable, Optional
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class GapAnalysisCacheMiddleware(BaseHTTPMiddleware):
    """
    Cache middleware for gap analysis endpoints.

    Caches GET requests to improve performance.
    """

    def __init__(self, app: Any, cache: Optional[Any] = None, ttl: int = 3600):
        """
        Initialize cache middleware.

        Args:
            app: FastAPI app
            cache: Cache client (e.g., Redis)
            ttl: Cache TTL in seconds
        """
        super().__init__(app)
        self.cache = cache
        self.ttl = ttl

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request with caching.

        Args:
            request: HTTP request
            call_next: Next middleware

        Returns:
            HTTP response
        """
        # Only cache GET requests
        if request.method != "GET" or self.cache is None:
            return await call_next(request)

        # Only cache gap endpoints
        if not request.url.path.startswith("/api/v1/gaps"):
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache
        try:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit: {cache_key}")
                return Response(
                    content=cached_response,
                    media_type="application/json",
                    headers={"X-Cache": "HIT"},
                )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

        # Process request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk

                # Cache the response
                self.cache.setex(cache_key, self.ttl, body)

                # Return response with new body
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        return response

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        # Include path and query params
        key_data = f"{request.url.path}:{request.url.query}"

        # Hash for shorter key
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return f"gap_cache:{key_hash}"