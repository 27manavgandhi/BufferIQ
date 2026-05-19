"""
Hashtag performance domain model.

Database model for hashtag performance tracking.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class HashtagPerformance(Base):
    """
    Hashtag performance model.

    Stores performance metrics for hashtags over time.
    """

    __tablename__ = "hashtag_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Hashtag info
    hashtag = Column(String(100), nullable=False, index=True)
    platform = Column(String(20), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)

    # Usage stats
    total_uses = Column(Integer, default=0)
    unique_posts = Column(Integer, default=0)
    first_used = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)

    # Engagement metrics
    avg_engagement = Column(Float, default=0.0)
    median_engagement = Column(Float, default=0.0)
    total_engagement = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)

    # Comparison
    engagement_lift = Column(Float, default=0.0)
    reach_amplification = Column(Float, default=0.0)

    # Distribution
    engagement_std = Column(Float, default=0.0)

    # Trend
    trend_direction = Column(String(20), default="stable")
    momentum = Column(Float, default=0.0)

    # ROI
    estimated_roi = Column(Float, default=0.0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_hashtag_platform", "hashtag", "platform"),
        Index("idx_hashtag_user", "hashtag", "user_id"),
        Index("idx_platform_updated", "platform", "updated_at"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<HashtagPerformance(hashtag='{self.hashtag}', platform='{self.platform}')>"