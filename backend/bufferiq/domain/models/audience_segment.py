"""Audience segment database model."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, JSON, Boolean, DateTime
from bufferiq.domain.base import Base


class AudienceSegment(Base):
    """Audience segment definition."""

    __tablename__ = "audience_segments"

    id = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    n_members = Column(Integer, nullable=False)
    size_percentage = Column(Float, nullable=False)
    centroid = Column(JSON)  # Cluster centroid vector

    # Clustering metadata
    clustering_algorithm = Column(String, nullable=False)
    silhouette_score = Column(Float)
    stability_score = Column(Float)

    # Status
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "n_members": self.n_members,
            "size_percentage": float(self.size_percentage),
            "clustering_algorithm": self.clustering_algorithm,
            "silhouette_score": float(self.silhouette_score) if self.silhouette_score else None,
            "stability_score": float(self.stability_score) if self.stability_score else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }