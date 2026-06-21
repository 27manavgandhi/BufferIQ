"""Cross-segment engagement analysis."""

from typing import Any, Dict, List

import numpy as np

from bufferiq.ml.segmentation.types import PersonaProfile


class CrossSegmentAnalyzer:
    """Analyze engagement across multiple segments."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize cross-segment analyzer."""
        self.config = config or {}

    def analyze(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """
        Analyze engagement patterns across segments.

        Args:
            personas: List of persona profiles

        Returns:
            Cross-segment analysis
        """
        if not personas:
            return self._empty_analysis()

        engagement_scores = [p.avg_engagement_rate for p in personas]
        growth_scores = [p.growth_potential_score for p in personas]
        retention_risks = [p.retention_risk_score for p in personas]

        return {
            "avg_engagement": float(np.mean(engagement_scores)),
            "engagement_variance": float(np.var(engagement_scores)),
            "high_engagement_segments": self._find_high_performers(personas),
            "low_engagement_segments": self._find_low_performers(personas),
            "growth_potential": float(np.mean(growth_scores)),
            "avg_retention_risk": float(np.mean(retention_risks)),
            "segment_count": len(personas),
        }

    def calculate_cross_impact(
        self,
        source_segment: PersonaProfile,
        target_segments: List[PersonaProfile],
    ) -> Dict[str, Any]:
        """
        Calculate impact of changes in one segment on others.

        Args:
            source_segment: Source segment
            target_segments: Target segments

        Returns:
            Cross-segment impact analysis
        """
        impacts = {}

        for target in target_segments:
            # Calculate similarity
            similarity = self._calculate_similarity(source_segment, target)

            # Calculate impact
            impact = similarity * 0.5  # 50% correlation

            impacts[target.segment_id] = {
                "similarity": similarity,
                "expected_impact": impact,
            }

        return impacts

    def _find_high_performers(self, personas: List[PersonaProfile]) -> List[str]:
        """Find high engagement segments."""
        threshold = np.mean([p.avg_engagement_rate for p in personas]) + np.std(
            [p.avg_engagement_rate for p in personas]
        )

        return [
            p.segment_id for p in personas if p.avg_engagement_rate > threshold
        ]

    def _find_low_performers(self, personas: List[PersonaProfile]) -> List[str]:
        """Find low engagement segments."""
        threshold = np.mean([p.avg_engagement_rate for p in personas]) - np.std(
            [p.avg_engagement_rate for p in personas]
        )

        return [
            p.segment_id for p in personas if p.avg_engagement_rate < threshold
        ]

    def _calculate_similarity(
        self, persona1: PersonaProfile, persona2: PersonaProfile
    ) -> float:
        """Calculate similarity between personas."""
        similarity = 0.0

        # Topic similarity
        topics1 = set(persona1.primary_topics)
        topics2 = set(persona2.primary_topics)
        if topics1 or topics2:
            topic_sim = len(topics1 & topics2) / len(topics1 | topics2)
            similarity += topic_sim * 0.3

        # Engagement similarity
        eng_diff = abs(persona1.avg_engagement_rate - persona2.avg_engagement_rate)
        eng_sim = 1.0 - min(eng_diff, 1.0)
        similarity += eng_sim * 0.4

        # Platform bonus
        if persona1.platform == persona2.platform:
            similarity += 0.3

        return float(min(similarity, 1.0))

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis."""
        return {
            "avg_engagement": 0.0,
            "engagement_variance": 0.0,
            "high_engagement_segments": [],
            "low_engagement_segments": [],
            "growth_potential": 0.0,
            "avg_retention_risk": 0.0,
            "segment_count": 0,
        }