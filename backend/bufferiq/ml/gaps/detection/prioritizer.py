"""Gap prioritization."""

from typing import List
import logging

from bufferiq.ml.gaps.detection.detector import ContentGap

logger = logging.getLogger(__name__)


class GapPrioritizer:
    """
    Prioritize content gaps for action.

    Ranks gaps by opportunity, effort, and strategic alignment.
    """

    def prioritize(self, gaps: List[ContentGap]) -> List[ContentGap]:
        """
        Prioritize gaps by multiple factors.

        Args:
            gaps: List of content gaps

        Returns:
            Sorted list of gaps by priority
        """
        # Calculate priority scores
        for gap in gaps:
            priority = self._calculate_priority(gap)
            gap.priority_score = round(priority, 2)

        # Sort by priority (descending)
        sorted_gaps = sorted(gaps, key=lambda g: g.priority_score, reverse=True)

        return sorted_gaps

    def _calculate_priority(self, gap: ContentGap) -> float:
        """
        Calculate priority score (0-100).

        Args:
            gap: Content gap

        Returns:
            Priority score
        """
        # Base score from opportunity
        score = gap.opportunity_score

        # Boost for trending topics
        if gap.trend_direction == "rising":
            score *= 1.3
        elif gap.trend_direction == "growing":
            score *= 1.15

        # Boost for high competitor coverage (validation)
        if gap.competitor_coverage >= 5:
            score *= 1.2
        elif gap.competitor_coverage >= 3:
            score *= 1.1

        # Boost for high estimated engagement
        if gap.estimated_engagement and gap.estimated_engagement > 200:
            score *= 1.1

        # Cap at 100
        score = min(score, 100)

        return score

    def identify_quick_wins(
        self, gaps: List[ContentGap], min_opportunity: float = 70
    ) -> List[ContentGap]:
        """
        Identify quick win opportunities.

        Args:
            gaps: List of gaps
            min_opportunity: Minimum opportunity score

        Returns:
            List of quick win gaps
        """
        quick_wins = [
            gap
            for gap in gaps
            if gap.opportunity_score >= min_opportunity and gap.priority_score >= 80
        ]

        # Sort by priority
        quick_wins.sort(key=lambda g: g.priority_score, reverse=True)

        return quick_wins[:10]  # Top 10