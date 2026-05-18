"""Gap severity classifier."""

from typing import List
import logging

from bufferiq.ml.gaps.detection.detector import ContentGap, GapSeverity

logger = logging.getLogger(__name__)


class GapClassifier:
    """
    Classify content gaps by severity.

    Uses multi-factor analysis to determine gap severity levels.
    """

    def __init__(
        self,
        critical_threshold: float = 80,
        important_threshold: float = 60,
        moderate_threshold: float = 40,
    ):
        """
        Initialize classifier.

        Args:
            critical_threshold: Threshold for critical gaps
            important_threshold: Threshold for important gaps
            moderate_threshold: Threshold for moderate gaps
        """
        self.critical_threshold = critical_threshold
        self.important_threshold = important_threshold
        self.moderate_threshold = moderate_threshold

    def classify(self, gaps: List[ContentGap]) -> List[ContentGap]:
        """
        Classify gaps by severity.

        Args:
            gaps: List of content gaps

        Returns:
            List of classified gaps
        """
        for gap in gaps:
            severity = self._classify_single(gap)
            gap.severity = severity

        return gaps

    def _classify_single(self, gap: ContentGap) -> GapSeverity:
        """
        Classify a single gap.

        Args:
            gap: Content gap to classify

        Returns:
            Gap severity
        """
        score = gap.opportunity_score

        # Adjust score based on trend
        if gap.trend_direction == "rising":
            score *= 1.2
        elif gap.trend_direction == "falling":
            score *= 0.8

        # Adjust based on competitor coverage
        if gap.competitor_coverage >= 5:
            score *= 1.1

        # Classify
        if score >= self.critical_threshold:
            return GapSeverity.CRITICAL
        elif score >= self.important_threshold:
            return GapSeverity.IMPORTANT
        elif score >= self.moderate_threshold:
            return GapSeverity.MODERATE
        else:
            return GapSeverity.MINOR

    def reclassify_by_context(
        self, gap: ContentGap, industry: str, user_goals: List[str]
    ) -> GapSeverity:
        """
        Reclassify gap based on additional context.

        Args:
            gap: Content gap
            industry: User's industry
            user_goals: User's content goals

        Returns:
            Updated severity
        """
        # Start with base classification
        severity = self._classify_single(gap)

        # Upgrade if topic aligns with user goals
        if any(goal.lower() in gap.topic.lower() for goal in user_goals):
            if severity == GapSeverity.IMPORTANT:
                severity = GapSeverity.CRITICAL
            elif severity == GapSeverity.MODERATE:
                severity = GapSeverity.IMPORTANT

        return severity