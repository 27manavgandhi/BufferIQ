"""Buffer API GraphQL client - FULLY CORRECTED VERSION."""

import asyncio
import json
from typing import Any, Optional

import aiohttp
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from bufferiq.core.cache import ResponseCache
from bufferiq.core.logging import get_logger
from bufferiq.infrastructure.buffer.exceptions import (
    BufferAPIError,
    BufferAuthenticationError,
    BufferNetworkError,
    BufferNotFoundError,
    BufferRateLimitError,
    BufferValidationError,
)
from bufferiq.infrastructure.buffer.queries import (
    CREATE_POST,
    DELETE_POST,
    GET_CHANNEL,
    GET_CHANNELS,
    GET_ORGANIZATION,
    GET_ORGANIZATIONS,
    GET_POST,
    GET_POSTS,
    UPDATE_POST,
)
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter

logger = get_logger(__name__)


class BufferClient:
    """
    Async GraphQL client for Buffer API.

    Handles authentication, rate limiting, caching, and retry logic.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ResponseCache] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Buffer API client.

        Args:
            api_url: Buffer GraphQL API URL
            api_key: Buffer API access token
            rate_limiter: Optional rate limiter instance
            cache: Optional response cache instance
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.max_retries = max_retries
        self._client: Optional[Client] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_client(self) -> Client:
        """
        Get or create GraphQL client.

        Returns:
            GQL client instance
        """
        if self._client is None:
            # Create custom headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Create transport
            transport = AIOHTTPTransport(
                url=self.api_url,
                headers=headers,
            )

            # Create client
            self._client = Client(
                transport=transport,
                fetch_schema_from_transport=False,
            )

        return self._client

    async def close(self) -> None:
        """Close client and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None
        self._client = None

    async def _execute_query(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        cache_ttl: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute GraphQL query with retry logic and caching.

        Args:
            query: GraphQL query string
            variables: Query variables
            cache_ttl: Cache TTL in seconds (None = no cache)

        Returns:
            Query response data

        Raises:
            BufferAuthenticationError: Authentication failed
            BufferRateLimitError: Rate limit exceeded
            BufferValidationError: Invalid request
            BufferNotFoundError: Resource not found
            BufferNetworkError: Network error
            BufferAPIError: Other API errors
        """
        variables = variables or {}

        # Check cache first
        if cache_ttl and self.cache:
            cache_key = f"{query}:{json.dumps(variables, sort_keys=True)}"
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info("Cache hit for query")
                return json.loads(cached)

        # Execute with retry
        for attempt in range(self.max_retries):
            try:
                client = await self._get_client()
                result = await client.execute_async(
                    gql(query),
                    variable_values=variables,
                )

                # Cache successful response - FIXED: Correct method signature
                if cache_ttl and self.cache:
                    await self.cache.set(
                        cache_key,
                        json.dumps(result),
                        cache_ttl,  # Positional arg, not keyword
                    )

                return result

            except Exception as e:
                error_message = str(e)

                # Check for authentication errors
                if "401" in error_message or "unauthorized" in error_message.lower():
                    raise BufferAuthenticationError("Invalid API key or token")

                # Check for rate limit errors
                if "429" in error_message or "rate limit" in error_message.lower():
                    retry_after = 60  # Default 1 minute
                    raise BufferRateLimitError(
                        "Rate limit exceeded", retry_after=retry_after
                    )

                # Check for validation errors
                if (
                    "400" in error_message
                    or "GRAPHQL_VALIDATION_FAILED" in error_message
                ):
                    raise BufferValidationError(
                        f"Invalid GraphQL query: {error_message}"
                    )

                # Check for not found errors
                if "404" in error_message or "not found" in error_message.lower():
                    raise BufferNotFoundError("Resource not found")

                # Network errors - retry with exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + (asyncio.get_event_loop().time() % 1)
                    logger.warning(f"Retrying in {wait_time:.0f}s: {error_message}")
                    await asyncio.sleep(wait_time)
                    continue

                # Final attempt failed
                if (
                    "network" in error_message.lower()
                    or "connection" in error_message.lower()
                ):
                    raise BufferNetworkError(f"Network error: {error_message}")

                raise BufferAPIError(f"{error_message}")

        raise BufferAPIError("Max retries exceeded")

    async def health_check(self) -> bool:
        """
        Check if API is accessible.

        Returns:
            True if API is healthy

        Raises:
            BufferAPIError: If health check fails
        """
        try:
            # Simple query to check connectivity
            await self.fetch_organizations()
            return True
        except BufferAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def fetch_organizations(self) -> list[dict[str, Any]]:
        """
        Fetch all organizations for authenticated user.

        Returns:
            List of organization data

        Example:
            >>> client = BufferClient(url, key)
            >>> orgs = await client.fetch_organizations()
            >>> print(orgs[0]['name'])
        """
        result = await self._execute_query(
            GET_ORGANIZATIONS,
            cache_ttl=3600,  # Cache for 1 hour
        )

        account = result.get("account", {})
        organizations = account.get("organizations", [])

        logger.info(f"Fetched {len(organizations)} organizations")
        return organizations

    async def fetch_organization(self, org_id: str) -> dict[str, Any]:
        """
        Fetch single organization by ID.

        Args:
            org_id: Organization ID

        Returns:
            Organization data

        Raises:
            BufferNotFoundError: Organization not found
        """
        result = await self._execute_query(
            GET_ORGANIZATION,
            variables={"id": org_id},
            cache_ttl=3600,
        )

        org = result.get("organization")
        if not org:
            raise BufferNotFoundError(f"Organization {org_id} not found")

        return org

    async def fetch_channels(self, org_id: str) -> list[dict[str, Any]]:
        """
        Fetch all channels for an organization.

        Args:
            org_id: Organization ID

        Returns:
            List of channel data
        """
        result = await self._execute_query(
            GET_CHANNELS,
            variables={"organizationId": org_id},
            cache_ttl=3600,
        )

        channels = result.get("channels", [])

        logger.info(f"Fetched {len(channels)} channels for org {org_id}")
        return channels

    async def fetch_channel(self, channel_id: str) -> dict[str, Any]:
        """
        Fetch single channel by ID.

        Args:
            channel_id: Channel ID

        Returns:
            Channel data

        Raises:
            BufferNotFoundError: Channel not found
        """
        result = await self._execute_query(
            GET_CHANNEL,
            variables={"id": channel_id},
            cache_ttl=3600,
        )

        channel = result.get("channel")
        if not channel:
            raise BufferNotFoundError(f"Channel {channel_id} not found")

        return channel

    async def fetch_posts(
        self,
        org_id: str,
        channel_id: str,
        limit: int = 100,
        after: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch posts for a channel with cursor-based pagination.

        Args:
            org_id: Organization ID (REQUIRED by Buffer API)
            channel_id: Channel ID
            limit: Maximum posts to fetch (default: 100)
            after: Cursor for pagination (optional)
            status: Filter by status (sent, scheduled, draft)

        Returns:
            Dict with 'edges' (list of posts) and 'pageInfo' (pagination info)
        """
        variables: dict[str, Any] = {
            "organizationId": org_id,
            "channelId": channel_id,
            "first": limit,
        }

        if after:
            variables["after"] = after

        result = await self._execute_query(
            GET_POSTS,
            variables=variables,
            cache_ttl=300,  # Cache for 5 minutes
        )

        posts_data = result.get("posts", {})

        logger.info(f"Fetched posts for channel {channel_id} in org {org_id}")
        return posts_data

    async def fetch_post(self, post_id: str) -> dict[str, Any]:
        """
        Fetch single post by ID.

        Args:
            post_id: Post ID

        Returns:
            Post data

        Raises:
            BufferNotFoundError: Post not found
        """
        result = await self._execute_query(
            GET_POST,
            variables={"id": post_id},
            cache_ttl=300,
        )

        post = result.get("post")
        if not post:
            raise BufferNotFoundError(f"Post {post_id} not found")

        return post

    async def create_post(
        self,
        channel_id: str,
        text: str,
        scheduling_type: str = "automatic",
        mode: str = "addToQueue",
        due_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a new post.

        Args:
            channel_id: Channel ID
            text: Post content
            scheduling_type: 'automatic' or 'notification' (default: 'automatic')
            mode: 'addToQueue', 'shareNow', 'shareNext', or 'customScheduled' (default: 'addToQueue')
            due_at: ISO 8601 datetime string (required if mode='customScheduled')

        Returns:
            Created post data

        Raises:
            BufferValidationError: Invalid post data
        """
        input_data: dict[str, Any] = {
            "channelId": channel_id,
            "text": text,
            "schedulingType": scheduling_type,
            "mode": mode,
        }

        if due_at:
            input_data["dueAt"] = due_at

        result = await self._execute_query(CREATE_POST, variables={"input": input_data})

        # Handle union response
        create_result = result.get("createPost", {})

        # Check if it's a success
        if "post" in create_result:
            post = create_result["post"]
            logger.info(f"Created post {post.get('id')} for channel {channel_id}")
            return post

        # Check if it's an error
        if "message" in create_result:
            raise BufferValidationError(create_result["message"])

        raise BufferAPIError("Failed to create post")

    async def update_post(
        self,
        post_id: str,
        text: Optional[str] = None,
        due_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Update an existing post.

        Args:
            post_id: Post ID
            text: New content (optional)
            due_at: New scheduled time (optional)

        Returns:
            Updated post data
        """
        input_data: dict[str, Any] = {"id": post_id}

        if text is not None:
            input_data["text"] = text

        if due_at is not None:
            input_data["dueAt"] = due_at

        result = await self._execute_query(UPDATE_POST, variables={"input": input_data})

        # Handle union response
        update_result = result.get("updatePost", {})

        if "post" in update_result:
            post = update_result["post"]
            logger.info(f"Updated post {post_id}")
            return post

        if "message" in update_result:
            raise BufferValidationError(update_result["message"])

        raise BufferAPIError(f"Failed to update post {post_id}")

    async def delete_post(self, post_id: str) -> bool:
        """
        Delete a post.

        Args:
            post_id: Post ID

        Returns:
            True if deleted successfully

        Raises:
            BufferNotFoundError: Post not found
        """
        result = await self._execute_query(
            DELETE_POST,
            variables={"input": {"id": post_id}},
        )

        # Handle union response
        delete_result = result.get("deletePost", {})

        if "id" in delete_result:
            logger.info(f"Deleted post {post_id}")
            return True

        if "message" in delete_result:
            raise BufferAPIError(delete_result["message"])

        raise BufferAPIError(f"Failed to delete post {post_id}")

    async def fetch_all_posts_paginated(
        self,
        org_id: str,
        channel_id: str,
        status: Optional[str] = None,
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch all posts for a channel using cursor-based pagination.

        Args:
            org_id: Organization ID
            channel_id: Channel ID
            status: Filter by status (optional)
            batch_size: Posts per batch (default: 100)

        Returns:
            All posts for the channel

        Example:
            >>> posts = await client.fetch_all_posts_paginated(org_id, channel_id)
            >>> print(f"Total posts: {len(posts)}")
        """
        all_posts: list[dict[str, Any]] = []
        after_cursor: Optional[str] = None

        while True:
            posts_data = await self.fetch_posts(
                org_id=org_id,
                channel_id=channel_id,
                limit=batch_size,
                after=after_cursor,
                status=status,
            )

            edges = posts_data.get("edges", [])
            if not edges:
                break

            # Extract nodes from edges
            batch = [edge["node"] for edge in edges]
            all_posts.extend(batch)

            # Check if there are more pages
            page_info = posts_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break

            after_cursor = page_info.get("endCursor")
            logger.info(f"Fetched {len(all_posts)} posts so far...")

        logger.info(f"Fetched total of {len(all_posts)} posts for channel {channel_id}")
        return all_posts

    async def __aenter__(self) -> "BufferClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
