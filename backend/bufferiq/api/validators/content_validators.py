"""
Content validation utilities.

Validation functions for content analysis.
"""

from typing import Any, Dict, List

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


def validate_platform(platform: str) -> None:
    """
    Validate platform is supported.

    Args:
        platform: Platform name

    Raises:
        ValueError: If platform not supported
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' not supported. "
            f"Supported: {SUPPORTED_PLATFORMS}"
        )


def validate_text(text: str) -> None:
    """
    Validate text content.

    Args:
        text: Text to validate

    Raises:
        ValueError: If text is invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if len(text) > 10000:
        raise ValueError("Text exceeds maximum length of 10000 characters")


def validate_batch_request(posts: List[Dict[str, Any]]) -> None:
    """
    Validate batch analysis request.

    Args:
        posts: List of posts

    Raises:
        ValueError: If request is invalid
    """
    if not posts:
        raise ValueError("Posts list cannot be empty")

    if len(posts) > 100:
        raise ValueError("Batch size cannot exceed 100 posts")

    for i, post in enumerate(posts):
        if "text" not in post:
            raise ValueError(f"Post {i} missing 'text' field")

        validate_text(post["text"])