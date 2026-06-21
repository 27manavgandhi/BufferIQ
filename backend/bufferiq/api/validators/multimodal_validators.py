"""Validators for multi-modal API requests."""

from typing import List
from fastapi import HTTPException


def validate_platform(platform: str) -> None:
    """
    Validate platform parameter.
    
    Args:
        platform: Platform name
        
    Raises:
        HTTPException: If platform not supported
    """
    supported = ["linkedin", "twitter", "bluesky"]
    
    if platform not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. Supported: {', '.join(supported)}"
        )


def validate_image_file(filename: str, content_type: str) -> None:
    """
    Validate image file.
    
    Args:
        filename: File name
        content_type: MIME type
        
    Raises:
        HTTPException: If file invalid
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Image type '{content_type}' not supported. Allowed: {', '.join(allowed_types)}"
        )
    
    # Check file extension
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"File extension not supported. Allowed: {', '.join(allowed_extensions)}"
        )


def validate_video_url(url: str) -> None:
    """
    Validate video URL.
    
    Args:
        url: Video URL
        
    Raises:
        HTTPException: If URL invalid
    """
    # Check URL scheme
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Video URL must use http:// or https:// scheme"
        )
    
    # Check file extension
    video_extensions = [".mp4", ".mov", ".avi", ".webm", ".mkv"]
    if not any(url.lower().endswith(ext) for ext in video_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Video format not supported. Allowed: {', '.join(video_extensions)}"
        )


def validate_urls_list(urls: List[str], max_count: int = 10) -> None:
    """
    Validate list of URLs.
    
    Args:
        urls: List of URLs
        max_count: Maximum allowed URLs
        
    Raises:
        HTTPException: If list invalid
    """
    if len(urls) > max_count:
        raise HTTPException(
            status_code=400,
            detail=f"Too many URLs. Maximum: {max_count}"
        )
    
    for url in urls:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL: {url}"
            )