"""Persona profile database model."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, JSON, Text, ForeignKey, DateTime
from bufferiq.domain.base import Base


class Persona(Base):
    """Persona profile for an audience segment."""

    __tablename__ = "personas"

    id = Column(String, primary_key=True)
    segment_id = Column(String, ForeignKey("audience_segments.id"), nullable=False, index=True)
    platform = Column(String, nullable=False, index=True)

    persona_name = Column(String, nullable=False)
    persona_description = Column(Text)

    # Demographics
    estimated_age_min = Column(Integer)
    estimated_age_max = Column(Integer)
    estimated_location = Column(String)
    verified_ratio = Column(Float)

    # Behavioral
    avg_engagement_rate = Column(Float)
    primary_interaction_type = Column(String)
    content_preferences = Column(JSON)
    peak_activity_hours = Column(JSON)
    peak_activity_days = Column(JSON)

    # Interests
    primary_topics = Column(JSON)
    secondary_topics = Column(JSON)
    avoided_topics = Column(JSON)

    # Scores
    engagement_potential_score = Column(Float)
    growth_potential_score = Column(Float)
    retention_risk_score = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "platform": self.platform,
            "persona_name": self.persona_name,
            "persona_description": self.persona_description,
            "estimated_age": (self.estimated_age_min, self.estimated_age_max),
            "estimated_location": self.estimated_location,
            "verified_ratio": float(self.verified_ratio) if self.verified_ratio else None,
            "avg_engagement_rate": float(self.avg_engagement_rate) if self.avg_engagement_rate else None,
            "engagement_potential_score": float(self.engagement_potential_score) if self.engagement_potential_score else None,
            "growth_potential_score": float(self.growth_potential_score) if self.growth_potential_score else None,
            "retention_risk_score": float(self.retention_risk_score) if self.retention_risk_score else None,
            "created_at": self.created_at.isoformat(),
        }