"""
Assignment engine.

Assigns users to experiment variants using deterministic
hash-based bucketing.

Key features:
    - Deterministic assignment
    - Consistent hashing
    - Assignment logging
    - Session tracking
    - Platform validation

Example:
```python
    engine = AssignmentEngine(db_session)
    
    assignment = engine.assign(
        experiment_config=config,
        user_id="user123",
        session_id="session456"
    )
    
    # Same user always gets same variant
    assignment2 = engine.assign(config, "user123")
    assert assignment.variant_id == assignment2.variant_id
```
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import hashlib

from sqlalchemy.orm import Session

from bufferiq.ml.experiments.design.designer import (
    ExperimentConfig,
    Variant,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.experiments.assignment.bucketing import HashBucketing
from bufferiq.ml.experiments.assignment.logger import AssignmentLogger


@dataclass
class Assignment:
    """Experiment assignment result."""

    experiment_id: str
    user_id: str
    variant_id: str
    variant_name: str

    # Assignment metadata
    assigned_at: datetime
    assignment_hash: str
    is_new_assignment: bool

    # Tracking
    session_id: Optional[str] = None
    platform: Optional[str] = None


class AssignmentEngine:
    """
    Assign users to experiment variants.

    Uses deterministic hash-based bucketing to ensure
    consistent assignments across sessions.

    Example:
```python
        engine = AssignmentEngine(db_session)

        assignment = engine.assign(
            experiment_config=config,
            user_id="user123",
            session_id="session456"
        )

        print(f"User assigned to: {assignment.variant_name}")
        print(f"Is new: {assignment.is_new_assignment}")

        # Same user, same experiment -> same variant
        assignment2 = engine.assign(
            experiment_config=config,
            user_id="user123"
        )

        assert assignment.variant_id == assignment2.variant_id
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize assignment engine.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self.assignment_logger = AssignmentLogger(db_session)
        self.bucketing = HashBucketing()

    def assign(
        self,
        experiment_config: ExperimentConfig,
        user_id: str,
        session_id: Optional[str] = None,
        platform: Optional[str] = None,
        force_variant: Optional[str] = None,
    ) -> Assignment:
        """
        Assign user to variant.

        Args:
            experiment_config: Experiment configuration
            user_id: User identifier
            session_id: Optional session ID
            platform: Optional platform
            force_variant: Optional forced variant (for testing)

        Returns:
            Assignment result

        Raises:
            ValueError: If platform not supported
        """
        # Validate platform
        if platform and platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        # Check for existing assignment
        existing = self.assignment_logger.get_assignment(
            experiment_id=experiment_config.experiment_id, user_id=user_id
        )

        if existing:
            return existing

        # Determine variant
        if force_variant:
            variant = self._get_variant_by_id(
                experiment_config.variants, force_variant
            )
        else:
            variant = self.bucketing.assign_variant(
                experiment_id=experiment_config.experiment_id,
                user_id=user_id,
                variants=experiment_config.variants,
            )

        # Create assignment
        assignment_hash = self._generate_hash(
            experiment_config.experiment_id, user_id, variant.id
        )

        assignment = Assignment(
            experiment_id=experiment_config.experiment_id,
            user_id=user_id,
            variant_id=variant.id,
            variant_name=variant.name,
            assigned_at=datetime.now(),
            assignment_hash=assignment_hash,
            is_new_assignment=True,
            session_id=session_id,
            platform=platform,
        )

        # Log assignment
        self.assignment_logger.log(assignment)

        return assignment

    def _get_variant_by_id(self, variants: list[Variant], variant_id: str) -> Variant:
        """
        Get variant by ID.

        Args:
            variants: List of variants
            variant_id: Variant ID

        Returns:
            Variant

        Raises:
            ValueError: If variant not found
        """
        for variant in variants:
            if variant.id == variant_id:
                return variant

        raise ValueError(f"Variant '{variant_id}' not found")

    def _generate_hash(
        self, experiment_id: str, user_id: str, variant_id: str
    ) -> str:
        """
        Generate assignment hash.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            variant_id: Variant ID

        Returns:
            SHA256 hash
        """
        data = f"{experiment_id}:{user_id}:{variant_id}"
        return hashlib.sha256(data.encode()).hexdigest()

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
        return self.assignment_logger.get_assignment(experiment_id, user_id)