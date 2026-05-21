"""
Experiment assignment domain model.

Database model for user assignments.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship

from bufferiq.domain.models.base import Base


class ExperimentAssignment(Base):
    """Experiment assignment database model."""

    __tablename__ = "experiment_assignments"
    __table_args__ = (
        Index("idx_experiment_user", "experiment_id", "user_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(255), ForeignKey("experiments.experiment_id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    variant_id = Column(String(100), nullable=False)
    variant_name = Column(String(500), nullable=False)
    
    # Assignment metadata
    assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    assignment_hash = Column(String(255), nullable=False)
    
    # Tracking
    session_id = Column(String(255), nullable=True)
    platform = Column(String(50), nullable=True)
    
    # Relationship
    experiment = relationship("Experiment", back_populates="assignments")

    def __repr__(self):
        return f"<ExperimentAssignment(experiment_id={self.experiment_id}, user_id={self.user_id}, variant={self.variant_id})>"