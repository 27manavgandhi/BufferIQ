"""
Dependencies for hashtag endpoints.

Provides dependency injection for hashtag services.
"""

from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from bufferiq.core.database import get_db
from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService
from bufferiq.api.services.hashtag_service import HashtagService


def get_hashtag_intelligence_service(
    db: Session = Depends(get_db),
) -> HashtagIntelligenceService:
    """
    Get hashtag intelligence service.

    Args:
        db: Database session

    Returns:
        Configured intelligence service
    """
    return HashtagIntelligenceService(db_session=db)


def get_hashtag_service(
    db: Session = Depends(get_db),
    intelligence: HashtagIntelligenceService = Depends(
        get_hashtag_intelligence_service
    ),
) -> HashtagService:
    """
    Get hashtag service.

    Args:
        db: Database session
        intelligence: Intelligence service

    Returns:
        Configured hashtag service
    """
    return HashtagService(
        db_session=db,
        intelligence_service=intelligence,
    )