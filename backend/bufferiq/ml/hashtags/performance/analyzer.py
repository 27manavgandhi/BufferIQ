"""
Hashtag performance analyzer.

Analyzes hashtag performance and impact on engagement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from scipy import stats
import numpy as np

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS


@dataclass
class HashtagPerformance:
    """Hashtag performance metrics."""

    hashtag: str
    platform: str

    # Usage stats
    total_uses: int
    unique_posts: int
    first_used: datetime
    last_used: datetime

    # Engagement metrics
    avg_engagement: float
    median_engagement: float
    total_engagement: int
    engagement_rate: float  # Engagement per use

    # Comparison
    engagement_lift: float  # vs posts without this hashtag
    reach_amplification: float  # Reach increase %

    # Distribution
    engagement_std: float
    engagement_percentiles: Dict[int, float] = field(default_factory=dict)

    # Trend
    trend_direction: str = "stable"  # "growing", "stable", "declining"
    momentum: float = 0.0  # Rate of growth/decline

    # ROI
    estimated_roi: float = 0.0  # Engagement gain per character used


@dataclass
class HashtagABTest:
    """A/B test result for hashtag."""

    hashtag: str
    platform: str

    # Groups
    with_hashtag: Dict[str, float]
    without_hashtag: Dict[str, float]

    # Statistical tests
    t_statistic: float
    p_value: float
    confidence_level: float

    # Results
    is_significant: bool
    effect_size: float  # Cohen's d
    recommendation: str


class HashtagPerformanceAnalyzer:
    """
    Analyze hashtag performance and impact.

    Calculates engagement metrics, compares performance,
    and runs statistical tests.

    Example:
```python
        analyzer = HashtagPerformanceAnalyzer(db_session)
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
            lookback_days=90
        )

        print(f"#{performance.hashtag}")
        print(f"  Uses: {performance.total_uses}")
        print(f"  Avg engagement: {performance.avg_engagement:.1f}")
        print(f"  Engagement lift: {performance.engagement_lift:.1%}")
        print(f"  Trend: {performance.trend_direction}")
        print(f"  ROI: {performance.estimated_roi:.2f}")
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize performance analyzer.

        Args:
            db_session: Database session
        """
        self.db = db_session

    async def analyze(
        self,
        hashtag: str,
        platform: str,
        lookback_days: int = 90,
        user_id: Optional[str] = None,
    ) -> HashtagPerformance:
        """
        Analyze hashtag performance.

        Args:
            hashtag: Hashtag to analyze (without #)
            platform: Platform name
            lookback_days: Days of history
            user_id: Optional user filter

        Returns:
            Performance metrics

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # This is a simplified implementation
        # In production, query actual database

        # Mock data for demonstration
        total_uses = 45
        unique_posts = 42
        first_used = datetime.now() - timedelta(days=lookback_days)
        last_used = datetime.now() - timedelta(days=1)

        # Engagement metrics (mock)
        engagements = np.random.normal(150, 30, total_uses)
        avg_engagement = float(np.mean(engagements))
        median_engagement = float(np.median(engagements))
        total_engagement = int(np.sum(engagements))
        engagement_rate = avg_engagement

        # Comparison (mock)
        engagement_lift = 0.25  # 25% lift
        reach_amplification = 0.35  # 35% more reach

        # Distribution
        engagement_std = float(np.std(engagements))
        engagement_percentiles = {
            25: float(np.percentile(engagements, 25)),
            50: float(np.percentile(engagements, 50)),
            75: float(np.percentile(engagements, 75)),
            90: float(np.percentile(engagements, 90)),
        }

        # Trend analysis
        trend_direction = "growing"
        momentum = 0.15  # 15% growth

        # ROI calculation
        estimated_roi = self.calculate_roi(
            avg_engagement_with=avg_engagement,
            avg_engagement_without=avg_engagement / (1 + engagement_lift),
            hashtag_length=len(hashtag),
        )

        return HashtagPerformance(
            hashtag=hashtag,
            platform=platform,
            total_uses=total_uses,
            unique_posts=unique_posts,
            first_used=first_used,
            last_used=last_used,
            avg_engagement=avg_engagement,
            median_engagement=median_engagement,
            total_engagement=total_engagement,
            engagement_rate=engagement_rate,
            engagement_lift=engagement_lift,
            reach_amplification=reach_amplification,
            engagement_std=engagement_std,
            engagement_percentiles=engagement_percentiles,
            trend_direction=trend_direction,
            momentum=momentum,
            estimated_roi=estimated_roi,
        )

    async def compare_with_without(
        self, hashtag: str, platform: str, user_id: str
    ) -> HashtagABTest:
        """
        Compare posts with vs without hashtag.

        Args:
            hashtag: Hashtag to test
            platform: Platform name
            user_id: User identifier

        Returns:
            A/B test results
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Mock A/B test data
        with_hashtag = {"avg_engagement": 150.0, "count": 30}
        without_hashtag = {"avg_engagement": 120.0, "count": 30}

        # Run t-test
        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1=with_hashtag["avg_engagement"],
            std1=25.0,
            nobs1=with_hashtag["count"],
            mean2=without_hashtag["avg_engagement"],
            std2=22.0,
            nobs2=without_hashtag["count"],
        )

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((25.0**2 + 22.0**2) / 2)
        effect_size = (
            with_hashtag["avg_engagement"] - without_hashtag["avg_engagement"]
        ) / pooled_std

        # Determine significance (p < 0.05)
        is_significant = p_value < 0.05
        confidence_level = 1 - p_value

        # Recommendation
        if is_significant and effect_size > 0.5:
            recommendation = "use"
        elif is_significant and effect_size > 0.2:
            recommendation = "use_with_caution"
        else:
            recommendation = "test_further"

        return HashtagABTest(
            hashtag=hashtag,
            platform=platform,
            with_hashtag=with_hashtag,
            without_hashtag=without_hashtag,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            confidence_level=float(confidence_level),
            is_significant=is_significant,
            effect_size=float(effect_size),
            recommendation=recommendation,
        )

    def calculate_roi(
        self,
        avg_engagement_with: float,
        avg_engagement_without: float,
        hashtag_length: int,
    ) -> float:
        """
        Calculate ROI per character.

        Args:
            avg_engagement_with: Avg engagement with hashtag
            avg_engagement_without: Avg engagement without
            hashtag_length: Character count of hashtag

        Returns:
            Engagement gain per character
        """
        engagement_gain = avg_engagement_with - avg_engagement_without
        # Add 1 for the # symbol
        total_chars = hashtag_length + 1
        return engagement_gain / total_chars if total_chars > 0 else 0.0