"""Priority matrix generation."""

from typing import Any, Dict, List
import logging

from bufferiq.ml.gaps.detection.detector import ContentGap

logger = logging.getLogger(__name__)


class PriorityMatrix:
    """
    Generate 2x2 priority matrix for content gaps.

    Classifies gaps into quadrants: Quick Wins, Strategic, Fill-ins, Low Priority
    """

    def generate(self, gaps: List[ContentGap]) -> Dict[str, List[ContentGap]]:
        """
        Generate priority matrix.

        Args:
            gaps: List of content gaps

        Returns:
            Dictionary with gaps organized by quadrant
        """
        quadrants = {
            "quick_wins": [],  # High value, low effort
            "strategic": [],  # High value, high effort
            "fill_ins": [],  # Low value, low effort
            "low_priority": [],  # Low value, high effort
        }

        for gap in gaps:
            quadrant = self._classify_quadrant(gap)
            quadrants[quadrant].append(gap)

        # Sort each quadrant by opportunity score
        for quadrant in quadrants:
            quadrants[quadrant].sort(
                key=lambda g: g.opportunity_score, reverse=True
            )

        return quadrants

    def _classify_quadrant(self, gap: ContentGap) -> str:
        """Classify gap into priority quadrant."""
        # Value axis: opportunity score
        high_value = gap.opportunity_score >= 70

        # Effort axis: based on competitor coverage and complexity
        high_effort = gap.competitor_coverage >= 5 or self._is_complex_topic(gap)

        if high_value and not high_effort:
            return "quick_wins"
        elif high_value and high_effort:
            return "strategic"
        elif not high_value and not high_effort:
            return "fill_ins"
        else:
            return "low_priority"

    def _is_complex_topic(self, gap: ContentGap) -> bool:
        """Determine if topic is complex."""
        # Simple heuristic: technical topics are more complex
        complex_keywords = [
            "architecture", "implementation", "advanced",
            "deep", "comprehensive", "framework"
        ]

        topic_lower = gap.topic.lower()
        return any(keyword in topic_lower for keyword in complex_keywords)

    def visualize_matrix(
        self, quadrants: Dict[str, List[ContentGap]]
    ) -> Dict[str, Any]:
        """
        Create visualization data for matrix.

        Args:
            quadrants: Priority quadrants

        Returns:
            Visualization data
        """
        matrix_data = []

        for quadrant_name, gaps in quadrants.items():
            for gap in gaps:
                # Calculate coordinates
                value = gap.opportunity_score
                effort = self._calculate_effort_score(gap)

                matrix_data.append({
                    "topic": gap.topic,
                    "quadrant": quadrant_name,
                    "value": value,
                    "effort": effort,
                    "priority_score": gap.priority_score,
                })

        return {
            "data_points": matrix_data,
            "quadrant_counts": {
                name: len(gaps) for name, gaps in quadrants.items()
            },
        }

    def _calculate_effort_score(self, gap: ContentGap) -> float:
        """Calculate effort score (0-100)."""
        # Based on multiple factors
        effort = 50.0

        # Competitor coverage increases effort
        effort += gap.competitor_coverage * 5

        # Complex topics increase effort
        if self._is_complex_topic(gap):
            effort += 20

        return min(effort, 100.0)