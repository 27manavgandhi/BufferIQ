"""Dependencies for multi-modal API endpoints."""

from fastapi import Depends
from sqlalchemy.orm import Session

from bufferiq.ml.multimodal.intelligence.service import MultiModalIntelligenceService
from bufferiq.api.services.multimodal_service import MultiModalService
from bufferiq.core.database import get_db


def get_intelligence_service() -> MultiModalIntelligenceService:
    """
    Get multi-modal intelligence service.
    
    Returns:
        MultiModalIntelligenceService instance
    """
    return MultiModalIntelligenceService()


def get_multimodal_service(
    db: Session = Depends(get_db),
    intelligence_service: MultiModalIntelligenceService = Depends(get_intelligence_service)
) -> MultiModalService:
    """
    Get multi-modal service with database.
    
    Args:
        db: Database session
        intelligence_service: Intelligence service
        
    Returns:
        MultiModalService instance
    """
    return MultiModalService(db, intelligence_service)