"""Visual features database model."""

from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, DateTime, Boolean, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime

from bufferiq.domain.models.base import Base


class VisualFeatures(Base):
    """Extracted visual features from media."""
    
    __tablename__ = "visual_features"
    
    id = Column(String, primary_key=True)
    media_analysis_id = Column(
        String,
        ForeignKey("media_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Image features
    objects_detected = Column(Integer, default=0)
    detected_objects = Column(JSON)  # List of detected objects with details
    text_extracted = Column(JSON)  # List of extracted text
    faces_detected = Column(Integer, default=0)
    face_details = Column(JSON)  # Face positions and emotions
    dominant_colors = Column(JSON)  # Color palette
    aesthetic_score = Column(Float)
    composition_scores = Column(JSON)  # Rule of thirds, golden ratio, etc.
    brand_elements = Column(JSON)  # Detected brand elements
    
    # Video features
    duration_seconds = Column(Float)
    video_resolution = Column(JSON)  # [width, height]
    fps = Column(Float)
    keyframe_count = Column(Integer)
    keyframes = Column(JSON)  # Keyframe details
    scene_count = Column(Integer)
    scenes = Column(JSON)  # Scene details
    has_audio = Column(Boolean, default=False)
    audio_features = Column(JSON)  # Audio analysis results
    
    # Embeddings (stored as JSON for compatibility)
    embedding_vector = Column(JSON)
    embedding_dimension = Column(Integer)
    
    # Quality metrics
    technical_quality = Column(Float)  # 0-100
    content_quality = Column(Float)  # 0-100
    engagement_potential = Column(Float)  # 0-1
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    media_analysis = relationship("MediaAnalysis", back_populates="visual_features")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VisualFeatures(id={self.id}, media_analysis_id={self.media_analysis_id})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "media_analysis_id": self.media_analysis_id,
            "objects_detected": self.objects_detected,
            "faces_detected": self.faces_detected,
            "aesthetic_score": self.aesthetic_score,
            "duration_seconds": self.duration_seconds,
            "keyframe_count": self.keyframe_count,
            "scene_count": self.scene_count,
            "has_audio": self.has_audio,
            "technical_quality": self.technical_quality,
            "content_quality": self.content_quality,
            "engagement_potential": self.engagement_potential,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }