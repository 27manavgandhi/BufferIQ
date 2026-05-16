"""
Voice service layer.

Business logic for voice API operations.
"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
import logging

from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService

logger = logging.getLogger(__name__)


class VoiceService:
    """
    Voice service for API operations.
    
    Provides business logic layer between API and
    voice intelligence service.
    """
    
    def __init__(self, db_session: Session, cache: Optional[Any] = None):
        """
        Initialize voice service.
        
        Args:
            db_session: Database session
            cache: Optional cache client
        """
        self.db = db_session
        self.intelligence = VoiceIntelligenceService(db_session, cache)
        logger.info("Voice service initialized")
    
    async def extract_profile(
        self, brand_id: str, platform: str, lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Extract voice profile.
        
        Args:
            brand_id: Brand identifier
            platform: Platform
            lookback_days: Days of history
        
        Returns:
            Profile data
        """
        profile = await self.intelligence.build_voice_profile(
            brand_id=brand_id,
            platform=platform,
            lookback_days=lookback_days,
        )
        
        return {
            "profile_id": profile.profile_id,
            "brand_id": profile.brand_id,
            "version": profile.version,
            "confidence": profile.confidence,
            "sample_size": profile.sample_size,
        }
    
    async def analyze_content(
        self,
        text: str,
        brand_id: str,
        platform: str,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze content voice.
        
        Args:
            text: Content to analyze
            brand_id: Brand identifier
            platform: Platform
            include_recommendations: Include recommendations
        
        Returns:
            Analysis results
        """
        return await self.intelligence.analyze_content(
            text=text,
            brand_id=brand_id,
            platform=platform,
            return_recommendations=include_recommendations,
        )