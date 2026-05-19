"""
Trend detector for hashtags.

Detects trending and emerging hashtags with lifecycle tracking.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS


class TrendStage(Enum):
    """Trend lifecycle stages."""

    EMERGING = "emerging"  # Just starting
    RISING = "rising"  # Rapid growth
    PEAK = "peak"  # Maximum popularity
    DECLINING = "declining"  # Losing momentum
    DORMANT = "dormant"  # No longer active


@dataclass
class TrendingHashtag:
    """Trending hashtag with metrics."""

    hashtag: str
    platform: str

    # Trend metrics
    stage: TrendStage
    momentum_score: float  # 0-100
    velocity: float  # Growth rate

    # Volume
    current_volume: int  # Uses in last 24h
    volume_change: float  # % change vs previous period
    peak_volume: int

    # Temporal
    trending_since: datetime
    time_to_peak: Optional[timedelta] = None
    predicted_expiration: Optional[datetime] = None

    # Context
    related_topics: List[str] = None  # type: ignore
    top_influencers: List[str] = None  # type: ignore
    geographic_hotspots: List[str] = None  # type: ignore

    # Opportunity
    opportunity_score: float = 0.0  # 0-100
    competition_level: str = "medium"  # "low", "medium", "high"
    recommendation: str = "monitor"  # "use_now", "monitor", "avoid"

    def __post_init__(self) -> None:
        """Initialize default lists."""
        if self.related_topics is None:
            self.related_topics = []
        if self.top_influencers is None:
            self.top_influencers = []
        if self.geographic_hotspots is None:
            self.geographic_hotspots = []


class TrendDetector:
    """
    Detect trending and emerging hashtags.

    Monitors hashtag volume, velocity, and momentum
    to identify trends in real-time.

    Example:
```python
        detector = TrendDetector(db_session)
        trends = await detector.detect_trending(
            platform="linkedin",
            category="technology",
            limit=20
        )

        for trend in trends[:10]:
            print(f"#{trend.hashtag}")
            print(f"  Stage: {trend.stage.value}")
            print(f"  Momentum: {trend.momentum_score:.1f}")
            print(f"  Volume change: {trend.volume_change:+.1%}")
            print(f"  Recommendation: {trend.recommendation}")
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize trend detector.

        Args:
            db_session: Database session
        """
        self.db = db_session

    async def detect_trending(
        self,
        platform: str,
        category: Optional[str] = None,
        limit: int = 50,
        min_volume: int = 100,
    ) -> List[TrendingHashtag]:
        """
        Detect currently trending hashtags.

        Args:
            platform: Platform to analyze
            category: Optional category filter
            limit: Maximum results
            min_volume: Minimum volume threshold

        Returns:
            List of trending hashtags

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Mock trending data for demonstration
        # In production, query real database

        trending: List[TrendingHashtag] = []

        # Example trending hashtags
        mock_trends = [
            {
                "hashtag": "ai",
                "current_volume": 1500,
                "previous_volume": 1000,
                "peak_volume": 1500,
                "stage": TrendStage.PEAK,
            },
            {
                "hashtag": "machinelearning",
                "current_volume": 1200,
                "previous_volume": 800,
                "peak_volume": 1300,
                "stage": TrendStage.RISING,
            },
            {
                "hashtag": "innovation",
                "current_volume": 800,
                "previous_volume": 900,
                "peak_volume": 1100,
                "stage": TrendStage.DECLINING,
            },
        ]

        for trend_data in mock_trends[:limit]:
            # Calculate metrics
            volume_change = (
                (trend_data["current_volume"] - trend_data["previous_volume"])
                / trend_data["previous_volume"]
                if trend_data["previous_volume"] > 0
                else 0.0
            )

            velocity = volume_change  # Simplified
            momentum = self.calculate_momentum(
                current_volume=trend_data["current_volume"],
                previous_volume=trend_data["previous_volume"],
                velocity=velocity,
            )

            # Determine opportunity and recommendation
            opportunity_score = self._calculate_opportunity_score(
                stage=trend_data["stage"],
                momentum=momentum,
                volume=trend_data["current_volume"],
            )

            competition_level = self._determine_competition_level(
                trend_data["current_volume"]
            )

            recommendation = self._get_recommendation(
                stage=trend_data["stage"],
                opportunity_score=opportunity_score,
                competition_level=competition_level,
            )

            trending_hashtag = TrendingHashtag(
                hashtag=trend_data["hashtag"],
                platform=platform,
                stage=trend_data["stage"],
                momentum_score=momentum,
                velocity=velocity,
                current_volume=trend_data["current_volume"],
                volume_change=volume_change,
                peak_volume=trend_data["peak_volume"],
                trending_since=datetime.now() - timedelta(days=7),
                opportunity_score=opportunity_score,
                competition_level=competition_level,
                recommendation=recommendation,
                related_topics=["technology", "business"],
                top_influencers=["@techinfluencer", "@aiexpert"],
                geographic_hotspots=["San Francisco", "New York"],
            )

            trending.append(trending_hashtag)

        # Sort by momentum score
        trending.sort(key=lambda x: x.momentum_score, reverse=True)

        return trending[:limit]

    def calculate_momentum(
        self, current_volume: int, previous_volume: int, velocity: float
    ) -> float:
        """
        Calculate trend momentum score (0-100).

        Args:
            current_volume: Current usage volume
            previous_volume: Previous period volume
            velocity: Growth velocity

        Returns:
            Momentum score
        """
        # Growth factor
        growth = (
            (current_volume - previous_volume) / previous_volume
            if previous_volume > 0
            else 0
        )

        # Combine volume and velocity (60% growth, 40% velocity)
        momentum = (growth * 0.6 + velocity * 0.4) * 100

        # Clamp to 0-100
        return max(0.0, min(100.0, momentum))

    def _calculate_opportunity_score(
        self, stage: TrendStage, momentum: float, volume: int
    ) -> float:
        """Calculate opportunity score (0-100)."""
        # Base score from momentum
        score = momentum * 0.5

        # Stage multiplier
        stage_multipliers = {
            TrendStage.EMERGING: 1.5,  # Best opportunity
            TrendStage.RISING: 1.2,
            TrendStage.PEAK: 0.8,
            TrendStage.DECLINING: 0.4,
            TrendStage.DORMANT: 0.1,
        }
        score *= stage_multipliers.get(stage, 1.0)

        # Volume factor (prefer moderate volume)
        if 500 <= volume <= 2000:
            score *= 1.2  # Sweet spot
        elif volume > 5000:
            score *= 0.7  # Too crowded

        return max(0.0, min(100.0, score))

    def _determine_competition_level(self, volume: int) -> str:
        """Determine competition level based on volume."""
        if volume < 500:
            return "low"
        elif volume < 2000:
            return "medium"
        else:
            return "high"

    def _get_recommendation(
        self, stage: TrendStage, opportunity_score: float, competition_level: str
    ) -> str:
        """Get usage recommendation."""
        if stage == TrendStage.EMERGING and opportunity_score > 70:
            return "use_now"
        elif stage == TrendStage.RISING and competition_level != "high":
            return "use_now"
        elif stage == TrendStage.PEAK and competition_level == "high":
            return "avoid"
        elif stage == TrendStage.DECLINING:
            return "avoid"
        else:
            return "monitor"