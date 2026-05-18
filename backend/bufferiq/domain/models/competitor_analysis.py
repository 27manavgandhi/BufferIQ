"""Competitor analysis domain model."""

from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CompetitorAnalysisModel(Base):
    """Competitor analysis database model."""

    __tablename__ = "competitor_analyses"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    platform = Column(String(20), nullable=False, index=True)

    # Analysis details
    competitor_ids = Column(JSON, nullable=False)  # List[str]
    analysis_period_days = Column(Integer, nullable=False)

    # Metrics
    user_rank = Column(Integer, nullable=False)
    share_of_voice = Column(Float, nullable=False)
    engagement_vs_avg = Column(Float, nullable=False)

    # Insights
    unique_topics = Column(JSON)  # List[str]
    missed_topics = Column(JSON)  # List[str]
    common_topics = Column(JSON)  # List[str]
    competitor_gaps = Column(JSON)  # List[str]
    differentiation_opportunities = Column(JSON)  # List[str]

    # Competitor profiles
    competitor_profiles = Column(JSON)  # List[Dict]

    # Metadata
    analyzed_at = Column(DateTime, default=datetime.now, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self) -> str:
        """String representation."""
        return f"<CompetitorAnalysis(id={self.id}, user={self.user_id}, rank={self.user_rank})>"