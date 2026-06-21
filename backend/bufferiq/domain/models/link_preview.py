"""Link preview database model."""

from sqlalchemy import Column, String, Float, JSON, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from bufferiq.domain.models.base import Base


class LinkPreview(Base):
    """Link preview analysis and metadata."""
    
    __tablename__ = "link_previews"
    
    id = Column(String, primary_key=True)
    media_analysis_id = Column(
        String,
        ForeignKey("media_analyses.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Metadata
    url = Column(Text, nullable=False)
    title = Column(Text)
    description = Column(Text)
    image_url = Column(Text)
    site_name = Column(String)
    
    # Tags
    og_tags = Column(JSON)  # Open Graph tags
    twitter_tags = Column(JSON)  # Twitter Card tags
    
    # Quality scores (0-100)
    title_quality = Column(Float)
    description_quality = Column(Float)
    image_quality = Column(Float)
    overall_quality = Column(Float)
    
    # CTR prediction
    predicted_ctr = Column(Float)
    actual_ctr = Column(Float)
    ctr_error = Column(Float)
    
    # Optimization
    optimization_suggestions = Column(JSON)  # List of suggestions
    optimized_title = Column(Text)
    optimized_description = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    media_analysis = relationship("MediaAnalysis", back_populates="link_preview")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<LinkPreview(id={self.id}, url={self.url[:50]})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "media_analysis_id": self.media_analysis_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "image_url": self.image_url,
            "site_name": self.site_name,
            "title_quality": self.title_quality,
            "description_quality": self.description_quality,
            "image_quality": self.image_quality,
            "overall_quality": self.overall_quality,
            "predicted_ctr": self.predicted_ctr,
            "actual_ctr": self.actual_ctr,
            "optimization_suggestions": self.optimization_suggestions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }