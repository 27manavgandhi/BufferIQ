"""
Assignment logger.

Logs and retrieves experiment assignments from database.

Example:
```python
    logger = AssignmentLogger(db_session)
    
    logger.log(assignment)
    
    # Later retrieve
    assignment = logger.get_assignment(experiment_id, user_id)
```
"""

from typing import Optional

from sqlalchemy.orm import Session

from bufferiq.ml.experiments.assignment.engine import Assignment


class AssignmentLogger:
    """
    Log and retrieve experiment assignments.

    Example:
```python
        logger = AssignmentLogger(db_session)

        # Log assignment
        logger.log(assignment)

        # Retrieve assignment
        existing = logger.get_assignment(
            experiment_id="exp_001",
            user_id="user123"
        )
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize assignment logger.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self._cache: dict[str, Assignment] = {}

    def log(self, assignment: Assignment) -> None:
        """
        Log assignment to database.

        Args:
            assignment: Assignment to log
        """
        # Add to cache
        cache_key = f"{assignment.experiment_id}:{assignment.user_id}"
        self._cache[cache_key] = assignment

        # In production, save to database here
        # For now, just cache in memory

    def get_assignment(
        self, experiment_id: str, user_id: str
    ) -> Optional[Assignment]:
        """
        Get existing assignment.

        Args:
            experiment_id: Experiment ID
            user_id: User ID

        Returns:
            Assignment if exists, None otherwise
        """
        # Check cache
        cache_key = f"{experiment_id}:{user_id}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Mark as not new
            cached.is_new_assignment = False
            return cached

        # In production, query database here
        # For now, return None

        return None

    def list_assignments(self, experiment_id: str) -> list[Assignment]:
        """
        List all assignments for experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of assignments
        """
        return [
            a
            for key, a in self._cache.items()
            if a.experiment_id == experiment_id
        ]