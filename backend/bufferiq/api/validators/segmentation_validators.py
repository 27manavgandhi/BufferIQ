"""Validators for segmentation API."""

from typing import List

from bufferiq.ml.segmentation.types import AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import ValidationError, UnsupportedPlatformError


def validate_platform(platform: str) -> None:
    """
    Validate platform.

    Args:
        platform: Platform to validate

    Raises:
        UnsupportedPlatformError: If platform not supported
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)


def validate_audience_data(audience_data: List[AudienceDataPoint]) -> None:
    """
    Validate audience data.

    Args:
        audience_data: Audience data to validate

    Raises:
        ValidationError: If data invalid
    """
    if not audience_data:
        raise ValidationError("audience_data cannot be empty")

    if len(audience_data) < 10:
        raise ValidationError("Need at least 10 audience members for segmentation")

    # Validate individual data points
    for item in audience_data:
        if not item.user_id:
            raise ValidationError("user_id cannot be empty")

        if item.follower_count < 0:
            raise ValidationError("follower_count cannot be negative")

        if item.following_count < 0:
            raise ValidationError("following_count cannot be negative")

        if not (0 <= item.avg_engagement_rate <= 1):
            raise ValidationError("avg_engagement_rate must be between 0 and 1")


def validate_segment_id(segment_id: str) -> None:
    """
    Validate segment ID.

    Args:
        segment_id: Segment ID to validate

    Raises:
        ValidationError: If invalid
    """
    if not segment_id:
        raise ValidationError("segment_id cannot be empty")

    if len(segment_id) > 100:
        raise ValidationError("segment_id cannot exceed 100 characters")