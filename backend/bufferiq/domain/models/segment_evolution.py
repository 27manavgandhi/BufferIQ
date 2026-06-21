"""Segment evolution tracking model."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, DateTime
from bufferiq.domain.base import Base


class SegmentEvolutionRecord(Base):
    """Historical record of segment evolution."""

    __tablename__ = "segment_evolution"

    id = Column(String, primary_key=True)
    segment_id = Column(String, ForeignKey("audience_segments.id"), nullable=False, index=True)
    platform = Column(String, nullable=False, index=True)

    snapshot_at = Column(DateTime, nullable=False, index=True)
    size = Column(Integer, nullable=False)
    avg_engagement_rate = Column(Float)
    health_score = Column(Float)
    centroid = Column(JSON)
    metrics = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "platform": self.platform,
            "snapshot_at": self.snapshot_at.isoformat(),
            "size": self.size,
            "avg_engagement_rate": float(self.avg_engagement_rate) if self.avg_engagement_rate else None,
            "health_score": float(self.health_score) if self.health_score else None,
            "created_at": self.created_at.isoformat(),
        }