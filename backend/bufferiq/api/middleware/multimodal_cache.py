"""Caching middleware for multi-modal analysis."""

from typing import Dict, Any, Optional
import hashlib
import json
from datetime import datetime, timedelta


class MultiModalCache:
    """Cache for multi-modal analysis results."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(
        self,
        media_url: str,
        platform: str,
        analysis_type: str
    ) -> str:
        """
        Generate cache key.
        
        Args:
            media_url: Media URL
            platform: Platform type
            analysis_type: Type of analysis
            
        Returns:
            Cache key
        """
        data = f"{media_url}:{platform}:{analysis_type}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(
        self,
        media_url: str,
        platform: str,
        analysis_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            media_url: Media URL
            platform: Platform type
            analysis_type: Type of analysis
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(media_url, platform, analysis_type)
        
        if key in self._cache:
            entry = self._cache[key]
            expires_at = entry["expires_at"]
            
            if datetime.now() < expires_at:
                return entry["data"]
            else:
                # Expired, remove from cache
                del self._cache[key]
        
        return None
    
    def set(
        self,
        media_url: str,
        platform: str,
        analysis_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Set cached result.
        
        Args:
            media_url: Media URL
            platform: Platform type
            analysis_type: Type of analysis
            data: Data to cache
        """
        key = self._generate_key(media_url, platform, analysis_type)
        
        self._cache[key] = {
            "data": data,
            "expires_at": datetime.now() + timedelta(seconds=self.ttl_seconds)
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now >= entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self._cache[key]