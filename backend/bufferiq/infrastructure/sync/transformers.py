"""
Transform Buffer API responses to database models.

Handles data validation, type conversion, and deduplication.
"""

import hashlib
from datetime import datetime
from typing import Any, Optional

from bufferiq.domain.models import Channel, Organization, Post


class BufferTransformer:
    """
    Transform Buffer API data to database models.

    Provides mapping and validation for all Buffer API entities.
    """

    @staticmethod
    def transform_organization(api_data: dict[str, Any], user_id: int) -> Organization:
        """
        Transform Buffer API organization to Organization model.

        Args:
            api_data: Organization data from Buffer API
            user_id: User ID who owns this organization

        Returns:
            Organization model instance

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["id", "name"]
        for field in required_fields:
            if field not in api_data:
                raise ValueError(f"Missing required field: {field}")

        return Organization(
            user_id=user_id,
            buffer_org_id=str(api_data["id"]),
            name=api_data["name"],
        )

    @staticmethod
    def transform_channel(api_data: dict[str, Any], organization_id: int) -> Channel:
        """
        Transform Buffer API channel to Channel model.

        Args:
            api_data: Channel data from Buffer API
            organization_id: Organization ID this channel belongs to

        Returns:
            Channel model instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ["id", "platform", "handle"]
        for field in required_fields:
            if field not in api_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate platform
        platform = api_data["platform"].lower()
        valid_platforms = {"linkedin", "twitter", "facebook", "instagram"}
        if platform not in valid_platforms:
            raise ValueError(f"Invalid platform: {platform}")

        return Channel(
            organization_id=organization_id,
            buffer_channel_id=str(api_data["id"]),
            platform=platform,
            handle=api_data["handle"],
            is_active=api_data.get("isActive", True),
        )

    @staticmethod
    def transform_post(api_data: dict[str, Any], channel_id: int) -> Post:
        """
        Transform Buffer API post to Post model.

        Args:
            api_data: Post data from Buffer API
            channel_id: Channel ID this post belongs to

        Returns:
            Post model instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ["id", "content", "status"]
        for field in required_fields:
            if field not in api_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate status
        status = api_data["status"].lower()
        valid_statuses = {"draft", "scheduled", "sent", "failed"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")

        # Generate content hash for deduplication
        content_hash = BufferTransformer.generate_content_hash(api_data["content"])

        # Extract engagement metrics
        engagement = api_data.get("engagement", {})
        likes = engagement.get("likes", 0)
        comments = engagement.get("comments", 0)
        shares = engagement.get("shares", 0)
        impressions = engagement.get("impressions", 0)

        # Calculate engagement rate
        engagement_rate = BufferTransformer.calculate_engagement_rate(
            likes, comments, shares, impressions
        )

        # Parse timestamps
        scheduled_at_str: Optional[str] = api_data.get("scheduledAt")
        scheduled_at = (
            BufferTransformer.parse_timestamp(scheduled_at_str)
            if scheduled_at_str is not None
            else None
        )
        sent_at_str: Optional[str] = api_data.get("sentAt")
        sent_at = (
            BufferTransformer.parse_timestamp(sent_at_str)
            if sent_at_str is not None
            else None
        )

        return Post(
            channel_id=channel_id,
            buffer_post_id=str(api_data["id"]),
            content=api_data["content"],
            content_hash=content_hash,
            status=status,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            likes=likes,
            comments=comments,
            shares=shares,
            impressions=impressions,
            engagement_rate=engagement_rate,
        )

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """
        Generate SHA256 hash of content for deduplication.

        Args:
            content: Post content text

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_engagement_rate(
        likes: int, comments: int, shares: int, impressions: int
    ) -> float:
        """
        Calculate engagement rate.

        Args:
            likes: Number of likes
            comments: Number of comments
            shares: Number of shares
            impressions: Number of impressions

        Returns:
            Engagement rate (0.0 to 1.0)
        """
        if impressions == 0:
            return 0.0

        total_engagement = likes + comments + shares
        return min(total_engagement / impressions, 1.0)

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> datetime:
        """
        Parse ISO 8601 timestamp to datetime.

        Args:
            timestamp_str: ISO 8601 timestamp string

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            # Try with timezone
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Try without timezone
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e
