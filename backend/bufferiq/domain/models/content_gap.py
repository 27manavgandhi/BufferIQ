"""Content gap domain model."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ContentGapModel(Base):
    """Content gap database model."""

    __tablename__ = "content_gaps"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    platform = Column(String(20), nullable=False, index=True)

    # Gap details
    topic = Column(String(200), nullable=False)
    keywords = Column(JSON, nullable=False)  # List[str]
    description = Column(String(500))

    # Scores
    severity = Column(String(20), nullable=False)  # critical/important/moderate/minor
    priority_score = Column(Float, nullable=False)
    opportunity_score = Column(Float, nullable=False)

    # Context
    competitor_coverage = Column(Integer, default=0)
    search_volume = Column(Integer, nullable=True)
    trend_direction = Column(String(20), default="stable")

    # Recommendations
    recommended_content_types = Column(JSON)  # List[str]
    suggested_angles = Column(JSON)  # List[str]
    estimated_engagement = Column(Float, nullable=True)

    # Metadata
    detected_at = Column(DateTime, default=datetime.now, nullable=False)
    confidence = Column(Float, default=0.8)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self) -> str:
        """String representation."""
        return f"<ContentGap(id={self.id}, topic={self.topic}, severity={self.severity})>"