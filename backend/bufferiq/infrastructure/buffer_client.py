"""
Buffer GraphQL API client with rate limiting, caching, and retry logic.
"""

import asyncio
import logging
import random
from typing import Any

import aiohttp

from bufferiq.core.cache import ResponseCache
from bufferiq.core.config import Settings
from bufferiq.infrastructure.buffer.exceptions import (
    BufferAPIError,
    BufferAuthenticationError,
    BufferNetworkError,
    BufferRateLimitError,
    BufferValidationError,
)
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class GraphQLResponse:
    """GraphQL response wrapper."""

    def __init__(
        self, data: dict[str, Any] | None, errors: list[dict[str, Any]] | None
    ) -> None:
        self.data = data
        self.errors = errors

    @property
    def has_errors(self) -> bool:
        """Check if response has errors."""
        return self.errors is not None and len(self.errors) > 0


class BufferClient:
    """
    Buffer GraphQL API client.

    Features:
    - Multi-tier rate limiting (100/15min, 500/24hr, 10K/30day)
    - Exponential backoff retry with jitter
    - Response caching with TTL
    - Comprehensive error handling
    """

    def __init__(
        self,
        settings: Settings,
        rate_limiter: RateLimiter,
        cache: ResponseCache,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 32.0,
    ) -> None:
        """
        Initialize Buffer client.

        Args:
            settings: Application settings
            rate_limiter: Rate limiter instance
            cache: Response cache instance
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self.settings = settings
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self.api_url = settings.buffer_api_url
        self.api_key = settings.buffer_api_key
        self.user_id = "default"  # Will be set from user context

    def set_user_id(self, user_id: str) -> None:
        """
        Set user ID for rate limiting.

        Args:
            user_id: User identifier
        """
        self.user_id = user_id

    def _get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers for API request.

        Returns:
            Headers dict
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "BufferIQ/1.0",
        }

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff with jitter.

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * 2^attempt
        delay = min(self.base_delay * (2**attempt), self.max_delay)

        # Add jitter (±25%)
        jitter = delay * 0.25
        delay_with_jitter = delay + random.uniform(-jitter, jitter)

        return float(max(0.1, delay_with_jitter))  # Minimum 100ms

    async def _execute_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Execute GraphQL request with retry logic.

        Args:
            query: GraphQL query/mutation
            variables: Query variables
            use_cache: Whether to use response cache

        Returns:
            Response data

        Raises:
            BufferAPIError: If request fails after all retries
        """
        # Check cache first (for queries only, not mutations)
        if use_cache and "mutation" not in query.lower():
            cached = await self.cache.get(query, variables)
            if cached is not None:
                logger.debug("Cache hit for query")
                return cached

        # Check rate limits
        await self.rate_limiter.check_limit(self.user_id)

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {"query": query, "variables": variables or {}}

                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=self._get_headers(),
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        # Increment rate limit counter
                        await self.rate_limiter.increment(self.user_id)

                        # Handle HTTP errors
                        if response.status == 401:
                            raise BufferAuthenticationError()

                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            retry_seconds = int(retry_after) if retry_after else None
                            raise BufferRateLimitError(retry_after=retry_seconds)

                        if response.status == 400:
                            error_data = await response.json()
                            raise BufferValidationError(
                                "Validation failed",
                                errors=error_data.get("errors", {}),
                            )

                        if response.status >= 500:
                            raise BufferAPIError(
                                f"Server error: {response.status}",
                                status_code=response.status,
                            )

                        # Parse response
                        data = await response.json()
                        graphql_response = GraphQLResponse(
                            data=data.get("data"),
                            errors=data.get("errors"),
                        )

                        # Check for GraphQL errors
                        if graphql_response.has_errors:
                            error_msg = "; ".join(
                                [
                                    e.get("message", "Unknown error")
                                    for e in graphql_response.errors or []
                                ]
                            )
                            raise BufferAPIError(f"GraphQL errors: {error_msg}")

                        # Cache successful response (queries only)
                        if use_cache and "mutation" not in query.lower():
                            await self.cache.set(
                                query, variables, graphql_response.data or {}
                            )

                        return graphql_response.data or {}

            except BufferAuthenticationError:
                # Don't retry auth errors
                raise

            except BufferValidationError:
                # Don't retry validation errors
                raise

            except (BufferAPIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = (
                    e
                    if isinstance(e, BufferAPIError)
                    else BufferNetworkError(
                        f"Network error: {str(e)}", original_error=e
                    )
                )

                # On last attempt, raise the error
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Request failed after {self.max_retries} retries: {last_error}"
                    )
                    break

                # Calculate backoff delay
                delay = self._calculate_delay(attempt)

                # Add extra delay for rate limit errors
                if isinstance(e, BufferRateLimitError) and e.retry_after:
                    delay = max(delay, float(e.retry_after))

                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        # All retries failed
        raise last_error or BufferAPIError("Request failed after all retries")

    async def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Execute GraphQL query.

        Args:
            query: GraphQL query
            variables: Query variables
            use_cache: Whether to use response cache

        Returns:
            Query result

        Raises:
            BufferAPIError: If query fails
        """
        return await self._execute_request(query, variables, use_cache)

    async def mutate(
        self,
        mutation: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute GraphQL mutation.

        Args:
            mutation: GraphQL mutation
            variables: Mutation variables

        Returns:
            Mutation result

        Raises:
            BufferAPIError: If mutation fails
        """
        # Mutations are never cached
        result = await self._execute_request(mutation, variables, use_cache=False)

        # Invalidate related cache entries
        await self.cache.clear()

        return result

    async def batch_query(
        self,
        queries: list[tuple[str, dict[str, Any] | None]],
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Execute multiple queries in parallel.

        Args:
            queries: List of (query, variables) tuples
            use_cache: Whether to use response cache

        Returns:
            List of query results in same order

        Raises:
            BufferAPIError: If any query fails
        """
        tasks = [self.query(q, v, use_cache) for q, v in queries]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """
        Check if API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple query to test connectivity
            query = "query { __typename }"
            await self.query(query, use_cache=False)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close client resources."""
        # Clean up any open connections
        # In this implementation, aiohttp sessions are closed automatically
        pass
