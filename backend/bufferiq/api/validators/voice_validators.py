"""
Voice API validators.

Input validation utilities for voice endpoints.
"""

from typing import List
import re


SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


def validate_platform(platform: str) -> None:
    """
    Validate platform is supported.
    
    Args:
        platform: Platform to validate
    
    Raises:
        ValueError: If platform not supported
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' not supported. "
            f"Supported platforms: {SUPPORTED_PLATFORMS}"
        )


def validate_text_content(text: str, min_length: int = 10, max_length: int = 10000) -> None:
    """
    Validate text content.
    
    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
    
    Raises:
        ValueError: If text invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) < min_length:
        raise ValueError(f"Text too short (minimum {min_length} characters)")
    
    if len(text) > max_length:
        raise ValueError(f"Text too long (maximum {max_length} characters)")


def validate_brand_id(brand_id: str) -> None:
    """
    Validate brand ID format.
    
    Args:
        brand_id: Brand ID to validate
    
    Raises:
        ValueError: If brand ID invalid
    """
    if not brand_id or not brand_id.strip():
        raise ValueError("Brand ID cannot be empty")
    
    # Allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', brand_id):
        raise ValueError("Brand ID contains invalid characters")


def validate_lookback_days(days: int) -> None:
    """
    Validate lookback days parameter.
    
    Args:
        days: Number of days
    
    Raises:
        ValueError: If days invalid
    """
    if days < 7:
        raise ValueError("Lookback days must be at least 7")
    
    if days > 365:
        raise ValueError("Lookback days cannot exceed 365")


def validate_batch_size(batch: List[str], max_size: int = 50) -> None:
    """
    Validate batch size.
    
    Args:
        batch: Batch of items
        max_size: Maximum batch size
    
    Raises:
        ValueError: If batch invalid
    """
    if not batch:
        raise ValueError("Batch cannot be empty")
    
    if len(batch) > max_size:
        raise ValueError(f"Batch size exceeds maximum ({max_size})")