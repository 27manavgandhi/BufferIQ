"""
Experiment repository.

Data access layer for experiments.
"""

from typing import List, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import and_

from bufferiq.domain.models.experiment import Experiment
from bufferiq.domain.models.experiment_result import ExperimentResult
from bufferiq.domain.models.experiment_assignment import ExperimentAssignment


class ExperimentRepository:
    """
    Repository for experiment data access.

    Example:
```python
        repo = ExperimentRepository(db_session)
        
        experiment = repo.create(experiment_data)
        found = repo.get_by_id(experiment.experiment_id)
```
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        self.db = db

    def create(self, experiment_data: dict) -> Experiment:
        """
        Create experiment.

        Args:
            experiment_data: Experiment data

        Returns:
            Created experiment
        """
        experiment = Experiment(**experiment_data)
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment

    def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment or None
        """
        return (
            self.db.query(Experiment)
            .filter(Experiment.experiment_id == experiment_id)
            .first()
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """
        List all experiments.

        Args:
            limit: Maximum results
            offset: Offset

        Returns:
            List of experiments
        """
        return (
            self.db.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def update_status(self, experiment_id: str, status: str) -> Optional[Experiment]:
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status

        Returns:
            Updated experiment or None
        """
        experiment = self.get_by_id(experiment_id)
        if experiment:
            experiment.status = status
            experiment.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(experiment)
        return experiment

    def delete(self, experiment_id: str) -> bool:
        """
        Delete experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False otherwise
        """
        experiment = self.get_by_id(experiment_id)
        if experiment:
            self.db.delete(experiment)
            self.db.commit()
            return True
        return False

    # Assignment methods
    def create_assignment(self, assignment_data: dict) -> ExperimentAssignment:
        """Create assignment."""
        assignment = ExperimentAssignment(**assignment_data)
        self.db.add(assignment)
        self.db.commit()
        self.db.refresh(assignment)
        return assignment

    def get_assignment(
        self, experiment_id: str, user_id: str
    ) -> Optional[ExperimentAssignment]:
        """Get assignment."""
        return (
            self.db.query(ExperimentAssignment)
            .filter(
                and_(
                    ExperimentAssignment.experiment_id == experiment_id,
                    ExperimentAssignment.user_id == user_id,
                )
            )
            .first()
        )

    # Result methods
    def create_result(self, result_data: dict) -> ExperimentResult:
        """Create result."""
        result = ExperimentResult(**result_data)
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get_latest_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get latest result."""
        return (
            self.db.query(ExperimentResult)
            .filter(ExperimentResult.experiment_id == experiment_id)
            .order_by(ExperimentResult.analyzed_at.desc())
            .first()
        )