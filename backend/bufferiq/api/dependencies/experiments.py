"""
Experiment dependencies.

FastAPI dependencies for experiment endpoints.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from bufferiq.api.services.experiment_service import ExperimentService
from bufferiq.api.dependencies.database import get_db


def get_experiment_service(db: Session = Depends(get_db)) -> ExperimentService:
    """
    Get experiment service.

    Args:
        db: Database session

    Returns:
        Experiment service
    """
    return ExperimentService(db)