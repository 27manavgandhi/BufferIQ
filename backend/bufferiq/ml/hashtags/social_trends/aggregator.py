"""
Cross-platform trend aggregator.

Aggregates trends across multiple platforms.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime


@dataclass
class CrossPlatformTrend:
    """Trend appearing across platforms."""

    hashtag: str
    platforms: List[str]
    total_volume: int
    platform_volumes: Dict[str, int] = field(default_factory=dict)

    # Metrics
    cross_platform_score: float = 0.0  # 0-100
    velocity: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)


class TrendAggregator:
    """
    Aggregate trends across platforms.

    Example:
```python
        aggregator = TrendAggregator()

        cross_trends = aggregator.aggregate_trends(
            platform_trends={
                "linkedin": [("ai", 1000), ("tech", 800)],
                "twitter": [("ai", 1500), ("ml", 700)],
                "bluesky": [("ai", 500)]
            }
        )

        for trend in cross_trends:
            print(f"#{trend.hashtag} - Platforms: {trend.platforms}")
```
    """

    def __init__(self, min_platforms: int = 2) -> None:
        """
        Initialize trend aggregator.

        Args:
            min_platforms: Minimum platforms to be considered cross-platform
        """
        self.min_platforms = min_platforms

    def aggregate_trends(
        self,
        platform_trends: Dict[str, List[tuple[str, int]]],
    ) -> List[CrossPlatformTrend]:
        """
        Aggregate trends across platforms.

        Args:
            platform_trends: Map of platform -> [(hashtag, volume)]

        Returns:
            List of cross-platform trends
        """
        # Collect all hashtags and their platform volumes
        hashtag_data: Dict[str, Dict[str, int]] = {}

        for platform, trends in platform_trends.items():
            for hashtag, volume in trends:
                if hashtag not in hashtag_data:
                    hashtag_data[hashtag] = {}
                hashtag_data[hashtag][platform] = volume

        # Build cross-platform trends
        cross_trends: List[CrossPlatformTrend] = []

        for hashtag, platform_volumes in hashtag_data.items():
            platforms = list(platform_volumes.keys())

            # Must appear on minimum number of platforms
            if len(platforms) >= self.min_platforms:
                total_volume = sum(platform_volumes.values())

                # Calculate cross-platform score
                score = self._calculate_cross_platform_score(
                    platform_count=len(platforms),
                    total_platforms=len(platform_trends),
                    total_volume=total_volume,
                )

                trend = CrossPlatformTrend(
                    hashtag=hashtag,
                    platforms=platforms,
                    total_volume=total_volume,
                    platform_volumes=platform_volumes,
                    cross_platform_score=score,
                    velocity=0.0,  # Would be calculated from historical data
                )

                cross_trends.append(trend)

        # Sort by cross-platform score
        cross_trends.sort(key=lambda t: t.cross_platform_score, reverse=True)

        return cross_trends

    def _calculate_cross_platform_score(
        self,
        platform_count: int,
        total_platforms: int,
        total_volume: int,
    ) -> float:
        """
        Calculate cross-platform score (0-100).

        Args:
            platform_count: Number of platforms hashtag appears on
            total_platforms: Total platforms checked
            total_volume: Total volume across platforms

        Returns:
            Cross-platform score
        """
        # Platform coverage
        coverage = platform_count / total_platforms

        # Volume factor (normalize to 0-1, assume max 10000)
        volume_factor = min(1.0, total_volume / 10000.0)

        # Weighted combination
        score = (coverage * 0.6 + volume_factor * 0.4) * 100

        return score