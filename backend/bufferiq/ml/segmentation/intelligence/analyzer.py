"""Segmentation analysis and insights."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import PersonaProfile
from bufferiq.ml.segmentation.prediction.cross_segment import CrossSegmentAnalyzer


class SegmentationAnalyzer:
    """Analyze segmentation results."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize analyzer."""
        self.config = config or {}
        self.cross_analyzer = CrossSegmentAnalyzer(self.config.get("cross_segment", {}))

    def analyze_results(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """
        Analyze complete segmentation results.

        Args:
            personas: List of generated personas

        Returns:
            Analysis results
        """
        if not personas:
            return {}

        # Cross-segment analysis
        cross_analysis = self.cross_analyzer.analyze(personas)

        # Segment quality metrics
        quality_metrics = self._assess_quality(personas)

        # Actionability assessment
        actionability = self._assess_actionability(personas)

        # Risk assessment
        risks = self._assess_risks(personas)

        return {
            "cross_segment_analysis": cross_analysis,
            "quality_metrics": quality_metrics,
            "actionability": actionability,
            "risks": risks,
            "summary": self._generate_summary(personas),
        }

    def _assess_quality(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """Assess segmentation quality."""
        engagement_scores = [p.avg_engagement_rate for p in personas]

        return {
            "avg_engagement": sum(engagement_scores) / len(engagement_scores),
            "engagement_variance": self._calculate_variance(engagement_scores),
            "segment_balance": self._assess_balance(personas),
        }

    def _assess_actionability(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """Assess actionability of personas."""
        action_scores = []

        for persona in personas:
            # Score based on clarity and distinctiveness
            score = 0.0
            if persona.primary_topics:
                score += 0.3
            if persona.peak_activity_hours:
                score += 0.2
            if persona.recommended_content_types:
                score += 0.25
            if persona.engagement_potential_score > 50:
                score += 0.25

            action_scores.append(score)

        return {
            "avg_actionability_score": sum(action_scores) / len(action_scores),
            "actionable_segments": sum(
                1 for score in action_scores if score > 0.7
            ),
        }

    def _assess_risks(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """Assess retention and engagement risks."""
        retention_risks = [p.retention_risk_score for p in personas]
        high_risk_count = sum(1 for r in retention_risks if r > 70)

        return {
            "avg_retention_risk": sum(retention_risks) / len(retention_risks),
            "high_risk_segments": high_risk_count,
            "risk_level": "high" if high_risk_count > len(personas) * 0.3 else "medium",
        }

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance."""
        if not values:
            return 0.0
        avg = sum(values) / len(values)
        return sum((x - avg) ** 2 for x in values) / len(values)

    def _assess_balance(self, personas: List[PersonaProfile]) -> float:
        """Assess balance of segment sizes."""
        sizes = [p.size for p in personas]
        total = sum(sizes)

        if total == 0:
            return 0.0

        # Calculate if segments are well-balanced
        expected_size = total / len(personas)
        variance = sum((s - expected_size) ** 2 for s in sizes) / len(sizes)
        balance_score = 1.0 / (1.0 + variance)

        return float(balance_score)

    def _generate_summary(self, personas: List[PersonaProfile]) -> str:
        """Generate summary of segmentation."""
        if not personas:
            return "No segments generated"

        largest = max(personas, key=lambda p: p.size)
        highest_engagement = max(personas, key=lambda p: p.avg_engagement_rate)

        summary = (
            f"Segmentation created {len(personas)} audience segments. "
            f"The largest segment is '{largest.persona_name}' ({largest.size} members). "
            f"Highest engagement is from '{highest_engagement.persona_name}' "
            f"({highest_engagement.avg_engagement_rate:.1%})."
        )

        return summary