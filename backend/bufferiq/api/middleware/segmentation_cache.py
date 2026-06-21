"""Caching middleware for segmentation."""

from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import hashlib
import json


class SegmentationCache:
    """Cache segmentation results."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """
        Initialize cache.

        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[Any, datetime]] = {}

    def get(
        self, platform: str, audience_hash: str
    ) -> Optional[Any]:
        """
        Get cached result.

        Args:
            platform: Platform type
            audience_hash: Hash of audience data

        Returns:
            Cached result or None
        """
        cache_key = f"{platform}:{audience_hash}"

        if cache_key not in self.cache:
            return None

        result, timestamp = self.cache[cache_key]

        # Check if expired
        if datetime.utcnow() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self.cache[cache_key]
            return None

        return result

    def set(self, platform: str, audience_hash: str, result: Any) -> None:
        """
        Cache result.

        Args:
            platform: Platform type
            audience_hash: Hash of audience data
            result: Result to cache
        """
        cache_key = f"{platform}:{audience_hash}"
        self.cache[cache_key] = (result, datetime.utcnow())

    def clear(self, platform: str | None = None) -> None:
        """
        Clear cache.

        Args:
            platform: Platform to clear (None for all)
        """
        if platform is None:
            self.cache.clear()
        else:
            keys_to_delete = [k for k in self.cache.keys() if k.startswith(platform)]
            for key in keys_to_delete:
                del self.cache[key]

    def clear_expired(self) -> None:
        """Clear expired entries."""
        current_time = datetime.utcnow()
        expired_keys = [
            k
            for k, (_, timestamp) in self.cache.items()
            if current_time - timestamp > timedelta(seconds=self.ttl_seconds)
        ]

        for key in expired_keys:
            del self.cache[key]

    @staticmethod
    def hash_audience(audience_data: list[Dict[str, Any]]) -> str:
        """
        Hash audience data.

        Args:
            audience_data: Audience data list

        Returns:
            MD5 hash
        """
        data_str = json.dumps(
            [{"id": item.get("user_id"), "engagement": item.get("avg_engagement_rate")}
             for item in audience_data],
            sort_keys=True,
        )

        return hashlib.md5(data_str.encode()).hexdigest()