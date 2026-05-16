"""
Voice profile domain model.

SQLAlchemy model for voice profiles.
"""

from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text
from sqlalchemy.sql import func
from datetime import datetime

from bufferiq.core.database import Base


class VoiceProfileModel(Base):
    """
    Voice profile database model.
    
    Stores brand voice profiles with fingerprints and metadata.
    """
    
    __tablename__ = "voice_profiles"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Profile identification
    profile_id = Column(String(255), unique=True, nullable=False, index=True)
    brand_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    
    # Fingerprints (stored as JSON)
    lexical_fingerprint = Column(JSON, nullable=False)
    syntactic_fingerprint = Column(JSON, nullable=False)
    stylistic_fingerprint = Column(JSON, nullable=False)
    
    # Voice signature
    signature = Column(String(64), nullable=False, index=True)
    
    # Metadata
    confidence = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    platform_profiles = Column(JSON, nullable=True)
    
    # Evolution tracking
    previous_version_id = Column(String(255), nullable=True)
    drift_from_previous = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<VoiceProfile(profile_id={self.profile_id}, brand_id={self.brand_id}, platform={self.platform})>"


class VoiceAnalysisLogModel(Base):
    """
    Voice analysis log model.
    
    Stores history of voice analyses for tracking and auditing.
    """
    
    __tablename__ = "voice_analysis_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Analysis identification
    brand_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    profile_id = Column(String(255), nullable=False)
    
    # Content
    content_hash = Column(String(64), nullable=False, index=True)
    content_length = Column(Integer, nullable=False)
    
    # Scores
    overall_score = Column(Float, nullable=False)
    lexical_score = Column(Float, nullable=False)
    syntactic_score = Column(Float, nullable=False)
    stylistic_score = Column(Float, nullable=False)
    
    # Metrics
    cosine_similarity = Column(Float, nullable=True)
    kl_divergence = Column(Float, nullable=True)
    
    # Results
    is_consistent = Column(Integer, nullable=False)  # 0 or 1 (boolean)
    severity = Column(String(50), nullable=False)
    
    # Timestamp
    analyzed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<VoiceAnalysisLog(brand_id={self.brand_id}, score={self.overall_score}, analyzed_at={self.analyzed_at})>"


class VoiceDriftLogModel(Base):
    """
    Voice drift log model.
    
    Stores history of drift detection results.
    """
    
    __tablename__ = "voice_drift_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Drift identification
    brand_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    
    # Detection results
    drift_detected = Column(Integer, nullable=False)  # 0 or 1 (boolean)
    drift_score = Column(Float, nullable=False)
    drift_type = Column(String(50), nullable=False)
    severity = Column(String(50), nullable=False)
    
    # Affected dimensions (stored as JSON array)
    affected_dimensions = Column(JSON, nullable=True)
    
    # Statistical tests
    t_statistic = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Timestamp
    checked_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<VoiceDriftLog(brand_id={self.brand_id}, drift_detected={self.drift_detected}, checked_at={self.checked_at})>"