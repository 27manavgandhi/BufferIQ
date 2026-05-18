"""Multi-factor opportunity scoring."""

from typing import Any, Dict, List
import logging

import numpy as np

from bufferiq.ml.gaps.detection.detector import ContentGap

logger = logging.getLogger(__name__)


class OpportunityScorer:
    """
    Score content opportunities using multiple factors.

    Combines various signals to produce comprehensive opportunity scores.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
    ):
        """
        Initialize opportunity scorer.

        Args:
            weights: Custom weights for scoring factors
        """
        # Default weights
        self.weights = weights or {
            "competitor_coverage": 0.25,
            "search_volume": 0.20,
            "trend_momentum": 0.25,
            "engagement_potential": 0.15,
            "strategic_fit": 0.15,
        }

    def score(
        self,
        gap: ContentGap,
        user_context: Dict[str, Any] = None,
    ) -> float:
        """
        Calculate comprehensive opportunity score.

        Args:
            gap: Content gap to score
            user_context: Optional user context for personalization

        Returns:
            Opportunity score (0-100)
        """
        # Calculate component scores
        competitor_score = self._score_competitor_coverage(gap)
        search_score = self._score_search_volume(gap)
        trend_score = self._score_trend(gap)
        engagement_score = self._score_engagement_potential(gap)
        strategic_score = self._score_strategic_fit(gap, user_context)

        # Weighted combination
        total_score = (
            (competitor_score * self.weights["competitor_coverage"]) +
            (search_score * self.weights["search_volume"]) +
            (trend_score * self.weights["trend_momentum"]) +
            (engagement_score * self.weights["engagement_potential"]) +
            (strategic_score * self.weights["strategic_fit"])
        )

        return round(total_score, 2)

    def _score_competitor_coverage(self, gap: ContentGap) -> float:
        """Score based on competitor coverage (validation)."""
        coverage = gap.competitor_coverage

        # More competitors = more validation
        # But too many = too competitive
        if coverage == 0:
            return 30.0  # Unproven
        elif coverage <= 2:
            return 60.0  # Some validation
        elif coverage <= 5:
            return 90.0  # Strong validation
        else:
            return 70.0  # High competition

    def _score_search_volume(self, gap: ContentGap) -> float:
        """Score based on search volume."""
        if gap.search_volume is None:
            return 50.0  # Default

        volume = gap.search_volume

        # Log scale normalization
        import math
        if volume <= 0:
            return 0.0

        # 100 searches = 20, 10000 searches = 80
        score = min(math.log10(volume) * 20, 100)

        return round(score, 2)

    def _score_trend(self, gap: ContentGap) -> float:
        """Score based on trend direction."""
        trend_map = {
            "rising": 100.0,
            "growing": 80.0,
            "stable": 50.0,
            "falling": 20.0,
        }

        return trend_map.get(gap.trend_direction, 50.0)

    def _score_engagement_potential(self, gap: ContentGap) -> float:
        """Score based on estimated engagement."""
        if gap.estimated_engagement is None:
            return 50.0

        engagement = gap.estimated_engagement

        # Normalize to 0-100
        score = min(engagement / 5, 100)

        return round(score, 2)

    def _score_strategic_fit(
        self, gap: ContentGap, user_context: Dict[str, Any] = None
    ) -> float:
        """Score based on strategic fit with user's goals."""
        if user_context is None:
            return 50.0

        # Check alignment with user's focus areas
        user_topics = user_context.get("focus_topics", [])
        user_goals = user_context.get("goals", [])

        score = 50.0

        # Boost if topic aligns with user's focus
        if any(topic.lower() in gap.topic.lower() for topic in user_topics):
            score += 25.0

        # Boost if aligns with goals
        if any(goal.lower() in gap.topic.lower() for goal in user_goals):
            score += 25.0

        return min(score, 100.0)

    def batch_score(
        self,
        gaps: List[ContentGap],
        user_context: Dict[str, Any] = None,
    ) -> List[ContentGap]:
        """
        Score multiple gaps.

        Args:
            gaps: List of content gaps
            user_context: Optional user context

        Returns:
            Gaps with updated opportunity scores
        """
        for gap in gaps:
            gap.opportunity_score = self.score(gap, user_context)

        return gaps

    def calculate_roi_estimate(
        self, gap: ContentGap, effort_hours: float = 4.0
    ) -> Dict[str, float]:
        """
        Calculate estimated ROI.

        Args:
            gap: Content gap
            effort_hours: Estimated hours to create content

        Returns:
            ROI metrics
        """
        # Simplified ROI calculation
        # In production, would use more sophisticated models

        expected_engagement = gap.estimated_engagement or 100.0
        effort_score = 100 - (effort_hours * 10)  # Inverse of effort

        roi_score = (expected_engagement / effort_hours) * gap.opportunity_score / 100

        return {
            "roi_score": round(roi_score, 2),
            "expected_engagement": expected_engagement,
            "effort_hours": effort_hours,
            "effort_score": round(max(effort_score, 0), 2),
        }