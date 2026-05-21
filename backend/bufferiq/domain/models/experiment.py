"""
Experiment domain model.

Database model for experiments.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship

from bufferiq.domain.models.base import Base


class Experiment(Base):
    """Experiment database model."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    type = Column(String(50), nullable=False)  # ab_test, multivariate, etc.
    platform = Column(String(50), nullable=False)
    primary_metric = Column(String(100), nullable=False)
    
    # Variants (stored as JSON)
    variants = Column(JSON, nullable=False)
    
    # Statistical parameters
    alpha = Column(Float, nullable=False, default=0.05)
    power = Column(Float, nullable=False, default=0.80)
    mde = Column(Float, nullable=False, default=0.10)
    required_sample_size = Column(Integer, nullable=False)
    
    # Duration
    estimated_duration_days = Column(Integer, nullable=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False, default="draft")  # draft, running, stopped, completed
    
    # Advanced options
    enable_sequential_testing = Column(Boolean, default=False)
    enable_early_stopping = Column(Boolean, default=False)
    stratification_key = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    
    # Relationships
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")
    assignments = relationship("ExperimentAssignment", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Experiment(id={self.experiment_id}, name={self.name}, platform={self.platform})>"