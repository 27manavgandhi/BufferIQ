"""Segment evolution tracking."""

from typing import Any, Dict, List

import numpy as np
from scipy import stats

from bufferiq.ml.segmentation.types import SegmentSnapshot, SegmentEvolution, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import UnsupportedPlatformError


class SegmentEvolutionTracker:
    """
    Track how audience segments evolve over time.

    Monitors:
    - Segment size changes
    - Engagement rate trends
    - Member migrations between segments
    - Centroid drift
    - Health score trends
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize evolution tracker."""
        self.config = config or {}

    def track(
        self,
        current_snapshot: SegmentSnapshot,
        historical_snapshots: List[SegmentSnapshot],
        platform: str,
    ) -> SegmentEvolution:
        """
        Track segment evolution.

        Args:
            current_snapshot: Current segment state
            historical_snapshots: List of past snapshots (oldest first)
            platform: Platform type

        Returns:
            Segment evolution analysis

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        all_snapshots = historical_snapshots + [current_snapshot]

        # Compute growth rate
        growth_rate = self._compute_growth_rate(all_snapshots)

        # Identify engagement trend
        engagement_trend = self._identify_engagement_trend(all_snapshots)

        # Compute stability
        stability = self._compute_stability(all_snapshots)

        # Predict future state
        predicted_size = self._predict_size(all_snapshots, days=30)
        predicted_engagement = self._predict_engagement(all_snapshots, days=30)

        # Analyze migrations
        migrations = self._analyze_migrations(
            historical_snapshots[-1] if historical_snapshots else None,
            current_snapshot,
        )

        return SegmentEvolution(
            segment_id=current_snapshot.segment_id,
            platform=platform,
            snapshots=all_snapshots,
            growth_rate=growth_rate,
            engagement_trend=engagement_trend,
            stability_score=stability,
            predicted_size_30d=predicted_size,
            predicted_engagement_30d=predicted_engagement,
            migration_summary=migrations,
        )

    def _compute_growth_rate(self, snapshots: List[SegmentSnapshot]) -> float:
        """Compute segment size growth rate."""
        if len(snapshots) < 2:
            return 0.0

        sizes = [s.size for s in snapshots]
        if sizes[0] == 0:
            return 0.0

        growth = (sizes[-1] - sizes[0]) / sizes[0]
        return float(growth)

    def _identify_engagement_trend(self, snapshots: List[SegmentSnapshot]) -> str:
        """Identify engagement rate trend using linear regression."""
        if len(snapshots) < 2:
            return "stable"

        engagement_rates = [s.avg_engagement_rate for s in snapshots]
        x = np.arange(len(engagement_rates))

        slope, _, r_value, p_value, _ = stats.linregress(x, engagement_rates)

        if p_value > 0.05:
            return "stable"
        elif slope > 0.001:
            return "improving"
        else:
            return "declining"

    def _compute_stability(self, snapshots: List[SegmentSnapshot]) -> float:
        """Compute segment stability score."""
        if len(snapshots) < 2:
            return 1.0

        sizes = [s.size for s in snapshots]
        if len(sizes) == 0 or np.mean(sizes) == 0:
            return 0.0

        # Calculate coefficient of variation
        cv = np.std(sizes) / np.mean(sizes)
        stability = 1.0 / (1.0 + cv)

        return float(stability)

    def _predict_size(
        self, snapshots: List[SegmentSnapshot], days: int = 30
    ) -> int:
        """Predict segment size in future days."""
        if len(snapshots) < 2:
            return snapshots[-1].size if snapshots else 0

        sizes = [s.size for s in snapshots[-10:]]  # Use last 10 snapshots
        x = np.arange(len(sizes))

        # Linear regression
        slope, intercept = np.polyfit(x, sizes, 1)

        # Predict
        future_x = x[-1] + (days / 30.0 * len(sizes))
        predicted = int(slope * future_x + intercept)

        return max(predicted, 1)

    def _predict_engagement(
        self, snapshots: List[SegmentSnapshot], days: int = 30
    ) -> float:
        """Predict engagement rate in future days."""
        if len(snapshots) < 2:
            return snapshots[-1].avg_engagement_rate if snapshots else 0.5

        engagement_rates = [s.avg_engagement_rate for s in snapshots[-10:]]
        x = np.arange(len(engagement_rates))

        slope, intercept = np.polyfit(x, engagement_rates, 1)
        future_x = x[-1] + (days / 30.0 * len(engagement_rates))
        predicted = slope * future_x + intercept

        return float(np.clip(predicted, 0.0, 1.0))

    def _analyze_migrations(
        self,
        previous_snapshot: SegmentSnapshot | None,
        current_snapshot: SegmentSnapshot,
    ) -> Dict[str, int]:
        """Analyze member migrations between snapshots."""
        if previous_snapshot is None:
            return {"inflow": 0, "outflow": 0}

        previous_members = set(previous_snapshot.member_ids)
        current_members = set(current_snapshot.member_ids)

        inflow = len(current_members - previous_members)
        outflow = len(previous_members - current_members)

        return {"inflow": inflow, "outflow": outflow}