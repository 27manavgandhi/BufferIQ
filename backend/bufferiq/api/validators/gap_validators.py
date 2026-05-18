"""Validators for gap analysis API."""

from typing import List
import re


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format.

    Args:
        user_id: User identifier

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    if not user_id or len(user_id) < 3:
        raise ValueError("User ID must be at least 3 characters")

    if len(user_id) > 100:
        raise ValueError("User ID must be less than 100 characters")

    # Alphanumeric and underscore only
    if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
        raise ValueError("User ID must be alphanumeric with underscore/hyphen only")

    return True


def validate_platform(platform: str) -> bool:
    """
    Validate platform.

    Args:
        platform: Platform name

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    allowed = ["linkedin", "twitter", "bluesky"]

    if platform not in allowed:
        raise ValueError(f"Platform must be one of: {allowed}")

    return True


def validate_competitor_ids(competitor_ids: List[str]) -> bool:
    """
    Validate competitor ID list.

    Args:
        competitor_ids: List of competitor IDs

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    if not competitor_ids:
        raise ValueError("At least one competitor ID required")

    if len(competitor_ids) > 10:
        raise ValueError("Maximum 10 competitors allowed")

    # Validate each ID
    for comp_id in competitor_ids:
        validate_user_id(comp_id)

    # Check for duplicates
    if len(competitor_ids) != len(set(competitor_ids)):
        raise ValueError("Duplicate competitor IDs not allowed")

    return True


def validate_lookback_days(days: int) -> bool:
    """
    Validate lookback days.

    Args:
        days: Number of days

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    if days < 7:
        raise ValueError("Lookback days must be at least 7")

    if days > 365:
        raise ValueError("Lookback days cannot exceed 365")

    return True