"""
Voice profile versioning.

Manages profile versions and tracks evolution over time.
"""

from typing import List, Optional, Dict
from datetime import datetime
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceProfileVersioning:
    """
    Manage voice profile versions.
    
    Tracks profile evolution and maintains version history.
    
    Example:
```python
        versioning = VoiceProfileVersioning()
        new_version = versioning.create_version(
            current_profile, new_features
        )
```
    """
    
    def __init__(self):
        """Initialize versioning manager."""
        self.version_history: Dict[str, List[VoiceProfile]] = {}
    
    def add_version(self, profile: VoiceProfile) -> None:
        """
        Add profile version to history.
        
        Args:
            profile: Voice profile to add
        """
        brand_id = profile.brand_id
        
        if brand_id not in self.version_history:
            self.version_history[brand_id] = []
        
        self.version_history[brand_id].append(profile)
        
        logger.info(
            f"Added voice profile version {profile.version} "
            f"for brand {brand_id}"
        )
    
    def get_latest_version(self, brand_id: str) -> Optional[VoiceProfile]:
        """
        Get latest profile version for brand.
        
        Args:
            brand_id: Brand identifier
        
        Returns:
            Latest profile or None
        """
        if brand_id not in self.version_history:
            return None
        
        versions = self.version_history[brand_id]
        if not versions:
            return None
        
        # Return most recent
        return max(versions, key=lambda p: p.version)
    
    def get_version_history(self, brand_id: str) -> List[VoiceProfile]:
        """
        Get complete version history for brand.
        
        Args:
            brand_id: Brand identifier
        
        Returns:
            List of profiles (oldest to newest)
        """
        if brand_id not in self.version_history:
            return []
        
        # Sort by version number
        return sorted(
            self.version_history[brand_id], key=lambda p: p.version
        )
    
    def calculate_total_drift(self, brand_id: str) -> float:
        """
        Calculate total drift across all versions.
        
        Args:
            brand_id: Brand identifier
        
        Returns:
            Total drift score
        """
        history = self.get_version_history(brand_id)
        
        if len(history) < 2:
            return 0.0
        
        total_drift = 0.0
        for i in range(1, len(history)):
            if history[i].drift_from_previous is not None:
                total_drift += history[i].drift_from_previous
        
        return total_drift
    
    def should_create_new_version(
        self, brand_id: str, drift_threshold: float = 0.15
    ) -> bool:
        """
        Determine if new version should be created based on drift.
        
        Args:
            brand_id: Brand identifier
            drift_threshold: Drift threshold for new version
        
        Returns:
            True if new version recommended
        """
        latest = self.get_latest_version(brand_id)
        
        if latest is None:
            return True
        
        # Check if significant drift from latest
        if latest.drift_from_previous is None:
            return False
        
        return latest.drift_from_previous > drift_threshold