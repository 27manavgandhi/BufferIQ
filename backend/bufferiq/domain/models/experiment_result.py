"""
Experiment result domain model.

Database model for experiment results.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship

from bufferiq.domain.models.base import Base


class ExperimentResult(Base):
    """Experiment result database model."""

    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(255), ForeignKey("experiments.experiment_id"), nullable=False, index=True)
    
    # Analysis metadata
    analyzed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    analysis_type = Column(String(50), nullable=False)  # standard, bayesian, sequential
    
    # Winner information
    has_winner = Column(Boolean, nullable=False, default=False)
    winner_variant = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    should_launch = Column(Boolean, default=False)
    
    # Statistical results
    test_type = Column(String(50), nullable=False)
    p_value = Column(Float, nullable=False)
    is_significant = Column(Boolean, nullable=False)
    effect_size = Column(Float, nullable=False)
    effect_size_type = Column(String(50), nullable=False)
    
    # Differences
    absolute_diff = Column(Float, nullable=False)
    relative_diff = Column(Float, nullable=False)
    
    # Confidence intervals
    ci_lower = Column(Float, nullable=False)
    ci_upper = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False, default=0.95)
    
    # Sample sizes
    n_control = Column(Integer, nullable=False)
    n_treatment = Column(Integer, nullable=False)
    
    # Means
    control_mean = Column(Float, nullable=False)
    treatment_mean = Column(Float, nullable=False)
    
    # Metrics breakdown
    control_metrics = Column(JSON, nullable=True)
    treatment_metrics = Column(JSON, nullable=True)
    
    # Recommendation
    recommendation = Column(Text, nullable=True)
    
    # Additional analysis
    segments = Column(JSON, nullable=True)
    time_series = Column(JSON, nullable=True)
    
    # Relationship
    experiment = relationship("Experiment", back_populates="results")

    def __repr__(self):
        return f"<ExperimentResult(experiment_id={self.experiment_id}, winner={self.winner_variant})>"