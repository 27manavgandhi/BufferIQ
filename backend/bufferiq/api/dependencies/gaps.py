"""Dependencies for gap analysis API."""

from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from bufferiq.api.services.gap_service import GapService
from bufferiq.core.database import get_db


def get_gap_service(
    db: Session = Depends(get_db),
) -> GapService:
    """
    Get gap service dependency.

    Args:
        db: Database session

    Returns:
        Gap service instance
    """
    return GapService(db_session=db)