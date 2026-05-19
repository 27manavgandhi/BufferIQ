"""
Hashtag repository.

Data access layer for hashtag operations.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from bufferiq.domain.models.hashtag_performance import HashtagPerformance
from bufferiq.domain.models.hashtag_trend import HashtagTrend


class HashtagRepository:
    """
    Repository for hashtag data operations.

    Provides data access methods for hashtag performance and trends.
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize repository.

        Args:
            db_session: Database session
        """
        self.db = db_session

    def get_performance(
        self,
        hashtag: str,
        platform: str,
        user_id: Optional[str] = None,
    ) -> Optional[HashtagPerformance]:
        """
        Get hashtag performance record.

        Args:
            hashtag: Hashtag to query
            platform: Platform name
            user_id: Optional user filter

        Returns:
            Performance record or None
        """
        query = self.db.query(HashtagPerformance).filter(
            and_(
                HashtagPerformance.hashtag == hashtag,
                HashtagPerformance.platform == platform,
            )
        )

        if user_id:
            query = query.filter(HashtagPerformance.user_id == user_id)

        return query.first()

    def save_performance(self, performance: HashtagPerformance) -> HashtagPerformance:
        """
        Save performance record.

        Args:
            performance: Performance record to save

        Returns:
            Saved record
        """
        self.db.add(performance)
        self.db.commit()
        self.db.refresh(performance)
        return performance

    def get_trending(
        self,
        platform: str,
        stage: Optional[str] = None,
        limit: int = 50,
        min_momentum: float = 0.0,
    ) -> List[HashtagTrend]:
        """
        Get trending hashtags.

        Args:
            platform: Platform to query
            stage: Optional stage filter
            limit: Maximum results
            min_momentum: Minimum momentum score

        Returns:
            List of trending hashtags
        """
        query = self.db.query(HashtagTrend).filter(
            and_(
                HashtagTrend.platform == platform,
                HashtagTrend.momentum_score >= min_momentum,
            )
        )

        if stage:
            query = query.filter(HashtagTrend.stage == stage)

        query = query.order_by(desc(HashtagTrend.momentum_score))

        return query.limit(limit).all()

    def save_trend(self, trend: HashtagTrend) -> HashtagTrend:
        """
        Save trend record.

        Args:
            trend: Trend record to save

        Returns:
            Saved record
        """
        self.db.add(trend)
        self.db.commit()
        self.db.refresh(trend)
        return trend

    def get_recent_performance(
        self,
        platform: str,
        days: int = 7,
        limit: int = 100,
    ) -> List[HashtagPerformance]:
        """
        Get recent performance records.

        Args:
            platform: Platform to query
            days: Days to look back
            limit: Maximum results

        Returns:
            List of performance records
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = (
            self.db.query(HashtagPerformance)
            .filter(
                and_(
                    HashtagPerformance.platform == platform,
                    HashtagPerformance.updated_at >= cutoff,
                )
            )
            .order_by(desc(HashtagPerformance.updated_at))
        )

        return query.limit(limit).all()