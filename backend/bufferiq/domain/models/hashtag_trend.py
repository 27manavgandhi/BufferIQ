"""
Hashtag trend domain model.

Database model for trending hashtag tracking.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Index, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class HashtagTrend(Base):
    """
    Hashtag trend model.

    Stores trending hashtag data over time.
    """

    __tablename__ = "hashtag_trends"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Hashtag info
    hashtag = Column(String(100), nullable=False, index=True)
    platform = Column(String(20), nullable=False, index=True)

    # Trend metrics
    stage = Column(String(20), nullable=False)  # emerging, rising, peak, declining, dormant
    momentum_score = Column(Float, default=0.0)
    velocity = Column(Float, default=0.0)

    # Volume
    current_volume = Column(Integer, default=0)
    volume_change = Column(Float, default=0.0)
    peak_volume = Column(Integer, default=0)

    # Temporal
    trending_since = Column(DateTime, nullable=True)
    time_to_peak = Column(Integer, nullable=True)  # Days

    # Context (stored as JSON)
    related_topics = Column(JSON, nullable=True)
    top_influencers = Column(JSON, nullable=True)
    geographic_hotspots = Column(JSON, nullable=True)

    # Opportunity
    opportunity_score = Column(Float, default=0.0)
    competition_level = Column(String(20), default="medium")
    recommendation = Column(String(20), default="monitor")

    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_trend_hashtag_platform", "hashtag", "platform"),
        Index("idx_trend_stage", "stage"),
        Index("idx_trend_momentum", "momentum_score"),
        Index("idx_trend_detected", "detected_at"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<HashtagTrend(hashtag='{self.hashtag}', stage='{self.stage}', momentum={self.momentum_score:.1f})>"