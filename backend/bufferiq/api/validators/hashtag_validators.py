"""
Validators for hashtag endpoints.

Custom validation logic for hashtag operations.
"""

from typing import List
from pydantic import validator

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS


class HashtagValidators:
    """Common validators for hashtag operations."""

    @staticmethod
    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """
        Validate platform is supported.

        Args:
            v: Platform value

        Returns:
            Validated platform

        Raises:
            ValueError: If platform not supported
        """
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        return v

    @staticmethod
    @validator("hashtag", "seed_hashtag")
    def validate_hashtag(cls, v: str) -> str:
        """
        Validate and clean hashtag.

        Args:
            v: Hashtag value

        Returns:
            Cleaned hashtag

        Raises:
            ValueError: If hashtag invalid
        """
        # Remove # if present
        cleaned = v.lstrip("#").strip()

        if not cleaned:
            raise ValueError("Hashtag cannot be empty")

        if len(cleaned) > 100:
            raise ValueError("Hashtag too long (max 100 characters)")

        # Check for invalid characters
        if not cleaned.replace("_", "").isalnum():
            raise ValueError(
                "Hashtag contains invalid characters (only letters, numbers, _ allowed)"
            )

        return cleaned

    @staticmethod
    @validator("hashtags")
    def validate_hashtag_list(cls, v: List[str]) -> List[str]:
        """
        Validate list of hashtags.

        Args:
            v: List of hashtags

        Returns:
            Validated list

        Raises:
            ValueError: If list invalid
        """
        if not v:
            raise ValueError("Hashtag list cannot be empty")

        if len(v) > 20:
            raise ValueError("Too many hashtags (max 20)")

        # Clean each hashtag
        cleaned = [h.lstrip("#").strip() for h in v]

        # Check for empty
        if any(not h for h in cleaned):
            raise ValueError("Hashtag list contains empty values")

        return cleaned