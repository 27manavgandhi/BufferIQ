"""Media analysis database model."""

from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from bufferiq.domain.models.base import Base


class MediaAnalysis(Base):
    """Media analysis results stored in database."""
    
    __tablename__ = "media_analyses"
    
    id = Column(String, primary_key=True)
    post_id = Column(String, ForeignKey("posts.id"), nullable=False, index=True)
    media_type = Column(String, nullable=False)  # image, video, link
    media_url = Column(Text, nullable=False)
    platform = Column(String, nullable=False, index=True)
    
    # Analysis results (JSON)
    analysis_data = Column(JSON, nullable=False)
    
    # Engagement metrics
    predicted_engagement = Column(Float)
    actual_engagement = Column(Float)
    prediction_error = Column(Float)
    
    # Processing metadata
    processing_time_ms = Column(Float)
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzer_version = Column(String, default="1.0.0")
    
    # Status
    status = Column(String, default="completed")  # completed, failed, pending
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    visual_features = relationship(
        "VisualFeatures",
        back_populates="media_analysis",
        cascade="all, delete-orphan"
    )
    link_preview = relationship(
        "LinkPreview",
        back_populates="media_analysis",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<MediaAnalysis(id={self.id}, type={self.media_type}, platform={self.platform})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "media_type": self.media_type,
            "media_url": self.media_url,
            "platform": self.platform,
            "analysis_data": self.analysis_data,
            "predicted_engagement": self.predicted_engagement,
            "actual_engagement": self.actual_engagement,
            "processing_time_ms": self.processing_time_ms,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }