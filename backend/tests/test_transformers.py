"""
Tests for data transformers.
"""

from datetime import datetime

import pytest

from bufferiq.infrastructure.sync.transformers import BufferTransformer


class TestBufferTransformer:
    """Test Buffer API data transformers."""

    def test_transform_organization(self) -> None:
        """Should transform organization data correctly."""
        api_data = {"id": "org_123", "name": "Test Organization"}

        org = BufferTransformer.transform_organization(api_data, user_id=1)

        assert org.user_id == 1
        assert org.buffer_org_id == "org_123"
        assert org.name == "Test Organization"

    def test_transform_organization_missing_field(self) -> None:
        """Should raise error on missing required field."""
        api_data = {"id": "org_123"}  # Missing name

        with pytest.raises(ValueError, match="Missing required field"):
            BufferTransformer.transform_organization(api_data, user_id=1)

    def test_transform_channel(self) -> None:
        """Should transform channel data correctly."""
        api_data = {
            "id": "ch_456",
            "platform": "LinkedIn",
            "handle": "testcompany",
            "isActive": True,
        }

        channel = BufferTransformer.transform_channel(api_data, organization_id=1)

        assert channel.organization_id == 1
        assert channel.buffer_channel_id == "ch_456"
        assert channel.platform == "linkedin"
        assert channel.handle == "testcompany"
        assert channel.is_active is True

    def test_transform_channel_invalid_platform(self) -> None:
        """Should raise error on invalid platform."""
        api_data = {"id": "ch_456", "platform": "invalid", "handle": "test"}

        with pytest.raises(ValueError, match="Invalid platform"):
            BufferTransformer.transform_channel(api_data, organization_id=1)

    def test_transform_post(self) -> None:
        """Should transform post data correctly."""
        api_data = {
            "id": "post_789",
            "content": "Test post content",
            "status": "sent",
            "scheduledAt": "2024-01-01T12:00:00Z",
            "sentAt": "2024-01-01T12:00:05Z",
            "engagement": {
                "likes": 10,
                "comments": 5,
                "shares": 2,
                "impressions": 1000,
            },
        }

        post = BufferTransformer.transform_post(api_data, channel_id=1)

        assert post.channel_id == 1
        assert post.buffer_post_id == "post_789"
        assert post.content == "Test post content"
        assert post.status == "sent"
        assert post.likes == 10
        assert post.comments == 5
        assert post.shares == 2
        assert post.impressions == 1000
        assert post.engagement_rate == 0.017  # (10+5+2)/1000

    def test_transform_post_no_engagement(self) -> None:
        """Should handle post with no engagement data."""
        api_data = {"id": "post_789", "content": "Test", "status": "draft"}

        post = BufferTransformer.transform_post(api_data, channel_id=1)

        assert post.likes == 0
        assert post.comments == 0
        assert post.shares == 0
        assert post.impressions == 0
        assert post.engagement_rate == 0.0

    def test_generate_content_hash(self) -> None:
        """Should generate consistent hash for content."""
        content = "Test post content"

        hash1 = BufferTransformer.generate_content_hash(content)
        hash2 = BufferTransformer.generate_content_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_calculate_engagement_rate(self) -> None:
        """Should calculate engagement rate correctly."""
        rate = BufferTransformer.calculate_engagement_rate(10, 5, 2, 1000)
        assert rate == 0.017

        # Zero impressions
        rate = BufferTransformer.calculate_engagement_rate(10, 5, 2, 0)
        assert rate == 0.0

        # Cap at 1.0
        rate = BufferTransformer.calculate_engagement_rate(1000, 500, 200, 1000)
        assert rate == 1.0

    def test_parse_timestamp(self) -> None:
        """Should parse ISO 8601 timestamps."""
        # With Z suffix
        ts = BufferTransformer.parse_timestamp("2024-01-01T12:00:00Z")
        assert isinstance(ts, datetime)
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 1

        # Without timezone
        ts = BufferTransformer.parse_timestamp("2024-01-01T12:00:00")
        assert isinstance(ts, datetime)

    def test_parse_invalid_timestamp(self) -> None:
        """Should raise error on invalid timestamp."""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            BufferTransformer.parse_timestamp("not-a-timestamp")
