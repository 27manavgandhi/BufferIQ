"""Type definitions for Buffer API."""

from typing import Any, Dict, List, Optional, TypedDict


class BufferProfile(TypedDict):
    """Buffer profile type."""

    id: str
    service: str
    service_username: str
    service_id: str


class BufferPost(TypedDict):
    """Buffer post type."""

    id: str
    profile_id: str
    text: str
    created_at: int
    due_at: Optional[int]
    statistics: Optional[Dict[str, Any]]


class BufferUpdate(TypedDict):
    """Buffer update type."""

    id: str
    profile_id: str
    text: str
    status: str
    created_at: int
    due_at: Optional[int]
    sent_at: Optional[int]
    statistics: Optional[Dict[str, int]]


class BufferStats(TypedDict):
    """Buffer statistics type."""

    reach: int
    clicks: int
    shares: int
    comments: int
    likes: int
    retweets: int
    favorites: int
    mentions: int


class BufferOrganization(TypedDict):
    """Buffer organization type."""

    id: str
    name: str
    created_at: int


__all__ = [
    "BufferProfile",
    "BufferPost",
    "BufferUpdate",
    "BufferStats",
    "BufferOrganization",
]