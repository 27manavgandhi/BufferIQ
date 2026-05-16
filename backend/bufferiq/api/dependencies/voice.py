"""
Voice API dependencies.

FastAPI dependency injection for voice endpoints.
"""

from fastapi import Depends
from sqlalchemy.orm import Session
from typing import Optional

from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService
from bufferiq.core.database import get_db


def get_voice_service(
    db: Session = Depends(get_db),
    cache: Optional[any] = None,
) -> VoiceIntelligenceService:
    """
    Get voice intelligence service.
    
    Args:
        db: Database session
        cache: Optional cache client
    
    Returns:
        Voice intelligence service
    """
    return VoiceIntelligenceService(db_session=db, cache=cache)