"""
Multi-brand voice profile management.

Manages voice profiles for multiple brands with
switching and comparison capabilities.
"""

from typing import Dict, List, Optional
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile

logger = logging.getLogger(__name__)


class MultiBrandVoiceManager:
    """
    Manage multiple brand voice profiles.
    
    Handles storage, retrieval, and switching between
    different brand voice profiles.
    
    Example:
```python
        manager = MultiBrandVoiceManager()
        manager.add_profile("brand_a", profile_a)
        manager.add_profile("brand_b", profile_b)
        
        active = manager.switch_brand("brand_b")
        all_brands = manager.list_brands()
```
    """
    
    def __init__(self):
        """Initialize multi-brand manager."""
        self.profiles: Dict[str, VoiceProfile] = {}
        self.active_brand: Optional[str] = None
    
    def add_profile(self, brand_id: str, profile: VoiceProfile) -> None:
        """
        Add voice profile for a brand.
        
        Args:
            brand_id: Brand identifier
            profile: Voice profile
        """
        self.profiles[brand_id] = profile
        logger.info(f"Added voice profile for brand {brand_id}")
        
        # Set as active if first brand
        if self.active_brand is None:
            self.active_brand = brand_id
    
    def get_profile(self, brand_id: str) -> Optional[VoiceProfile]:
        """
        Get voice profile for a brand.
        
        Args:
            brand_id: Brand identifier
        
        Returns:
            Voice profile or None if not found
        """
        return self.profiles.get(brand_id)
    
    def switch_brand(self, brand_id: str) -> VoiceProfile:
        """
        Switch to a different brand.
        
        Args:
            brand_id: Brand to switch to
        
        Returns:
            Active voice profile
        
        Raises:
            ValueError: If brand not found
        """
        if brand_id not in self.profiles:
            raise ValueError(f"Brand {brand_id} not found")
        
        self.active_brand = brand_id
        logger.info(f"Switched to brand {brand_id}")
        
        return self.profiles[brand_id]
    
    def get_active_profile(self) -> Optional[VoiceProfile]:
        """
        Get currently active voice profile.
        
        Returns:
            Active profile or None
        """
        if self.active_brand is None:
            return None
        
        return self.profiles.get(self.active_brand)
    
    def list_brands(self) -> List[str]:
        """
        List all managed brands.
        
        Returns:
            List of brand IDs
        """
        return list(self.profiles.keys())
    
    def remove_profile(self, brand_id: str) -> bool:
        """
        Remove voice profile for a brand.
        
        Args:
            brand_id: Brand to remove
        
        Returns:
            True if removed, False if not found
        """
        if brand_id in self.profiles:
            del self.profiles[brand_id]
            
            # Clear active if it was the removed brand
            if self.active_brand == brand_id:
                self.active_brand = None
            
            logger.info(f"Removed voice profile for brand {brand_id}")
            return True
        
        return False
    
    def get_all_profiles(self) -> Dict[str, VoiceProfile]:
        """
        Get all voice profiles.
        
        Returns:
            Dictionary of brand_id -> profile
        """
        return self.profiles.copy()