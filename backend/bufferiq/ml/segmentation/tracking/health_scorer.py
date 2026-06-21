"""Segment health scoring."""

from typing import Any, Dict, List

import numpy as np

from bufferiq.ml.segmentation.types import SegmentSnapshot


class HealthScorer:
    """Score and monitor segment health."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize health scorer."""
        self.config = config or {}

    def score_health(self, snapshot: SegmentSnapshot) -> float:
        """
        Score segment health (0-100).

        Args:
            snapshot: Segment snapshot

        Returns:
            Health score
        """
        score = 0.0

        # Size component (30%)
        size_score = self._score_size(snapshot.size)
        score += size_score * 0.3

        # Engagement component (40%)
        engagement_score = self._score_engagement(snapshot.avg_engagement_rate)
        score += engagement_score * 0.4

        # Stability component (30%)
        stability_score = snapshot.health_score if snapshot.health_score > 0 else 50.0
        score += stability_score * 0.3

        return min(max(score, 0.0), 100.0)

    def monitor_health_trend(
        self, snapshots: List[SegmentSnapshot]
    ) -> Dict[str, Any]:
        """
        Monitor health trend over time.

        Args:
            snapshots: List of snapshots

        Returns:
            Health trend analysis
        """
        if not snapshots:
            return {
                "current_health": 0.0,
                "trend": "unknown",
                "risk_level": "unknown",
            }

        health_scores = [self.score_health(s) for s in snapshots]

        current_health = health_scores[-1]
        trend = self._determine_trend(health_scores)
        risk_level = self._determine_risk(current_health, trend)

        return {
            "current_health": float(current_health),
            "trend": trend,
            "risk_level": risk_level,
            "health_history": health_scores,
        }

    def _score_size(self, size: int) -> float:
        """Score based on segment size."""
        if size < 10:
            return 20.0
        elif size < 50:
            return 50.0
        elif size < 100:
            return 80.0
        else:
            return 100.0

    def _score_engagement(self, engagement_rate: float) -> float:
        """Score based on engagement rate."""
        return engagement_rate * 100.0

    def _determine_trend(self, health_scores: List[float]) -> str:
        """Determine health trend."""
        if len(health_scores) < 2:
            return "unknown"

        recent_scores = health_scores[-5:]
        avg_recent = np.mean(recent_scores)
        avg_previous = np.mean(health_scores[:-5]) if len(health_scores) > 5 else np.mean(recent_scores[:-1])

        if avg_recent > avg_previous + 5:
            return "improving"
        elif avg_recent < avg_previous - 5:
            return "declining"
        else:
            return "stable"

    def _determine_risk(self, health_score: float, trend: str) -> str:
        """Determine risk level."""
        if health_score < 30:
            return "critical"
        elif health_score < 50:
            return "high"
        elif health_score < 70:
            if trend == "declining":
                return "medium"
            else:
                return "low"
        else:
            return "low"