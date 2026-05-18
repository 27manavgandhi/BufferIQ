"""Gap analysis service layer."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService

logger = logging.getLogger(__name__)


class GapService:
    """
    Service layer for gap analysis API.

    Wraps GapIntelligenceService with API-specific logic.
    """

    def __init__(self, db_session: Session, cache: Optional[Any] = None):
        """
        Initialize gap service.

        Args:
            db_session: Database session
            cache: Optional cache client
        """
        self.intelligence_service = GapIntelligenceService(
            db_session=db_session,
            cache=cache,
        )

    async def analyze_gaps(
        self,
        user_id: str,
        platform: str,
        competitor_ids: Optional[List[str]] = None,
        industry: Optional[str] = None,
        lookback_days: int = 90,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze content gaps.

        Args:
            user_id: User identifier
            platform: Platform to analyze
            competitor_ids: Optional competitors
            industry: Optional industry
            lookback_days: Days of history
            include_recommendations: Include recommendations

        Returns:
            Gap analysis report
        """
        logger.info(
            f"Gap analysis request: user={user_id}, platform={platform}"
        )

        report = await self.intelligence_service.analyze_gaps(
            user_id=user_id,
            platform=platform,
            competitor_ids=competitor_ids,
            industry=industry,
            lookback_days=lookback_days,
            include_recommendations=include_recommendations,
        )

        # Add API metadata
        report["api_version"] = "1.0"
        report["processing_time_ms"] = 0  # Would track actual time

        return report

    async def get_recommendations(
        self,
        user_id: str,
        platform: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get content recommendations.

        Args:
            user_id: User identifier
            platform: Platform
            count: Number of recommendations

        Returns:
            List of recommendations
        """
        logger.info(
            f"Recommendations request: user={user_id}, count={count}"
        )

        # Get gap analysis first
        report = await self.intelligence_service.analyze_gaps(
            user_id=user_id,
            platform=platform,
            include_recommendations=True,
        )

        recommendations = report.get("recommendations", [])

        return recommendations[:count]

    async def generate_calendar(
        self,
        user_id: str,
        platform: str,
        weeks: int = 4,
        posts_per_week: int = 3,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate content calendar.

        Args:
            user_id: User identifier
            platform: Platform
            weeks: Number of weeks
            posts_per_week: Posts per week
            start_date: Start date

        Returns:
            Content calendar
        """
        logger.info(
            f"Calendar request: user={user_id}, weeks={weeks}"
        )

        calendar = await self.intelligence_service.generate_calendar(
            user_id=user_id,
            platform=platform,
            weeks=weeks,
            posts_per_week=posts_per_week,
            start_date=start_date,
        )

        return calendar

    async def benchmark_competitors(
        self,
        user_id: str,
        competitor_ids: List[str],
        platform: str,
    ) -> Dict[str, Any]:
        """
        Benchmark against competitors.

        Args:
            user_id: User identifier
            competitor_ids: Competitor IDs
            platform: Platform

        Returns:
            Competitive analysis
        """
        logger.info(
            f"Competitor benchmark: user={user_id}, competitors={len(competitor_ids)}"
        )

        analysis = await self.intelligence_service.benchmark_competitors(
            user_id=user_id,
            competitor_ids=competitor_ids,
            platform=platform,
        )

        return analysis

    async def get_quick_insights(
        self,
        user_id: str,
        platform: str,
    ) -> Dict[str, Any]:
        """
        Get quick gap insights.

        Args:
            user_id: User identifier
            platform: Platform

        Returns:
            Quick insights
        """
        logger.info(f"Quick insights: user={user_id}")

        insights = await self.intelligence_service.get_quick_insights(
            user_id=user_id,
            platform=platform,
        )

        return insights