"""Competitor strategy pattern detection."""

from typing import Any, Dict, List
import logging
from collections import Counter

from bufferiq.ml.gaps.competitors.analyzer import CompetitorProfile

logger = logging.getLogger(__name__)


class StrategyDetector:
    """
    Detect competitor content strategy patterns.

    Identifies common strategies and successful patterns.
    """

    def detect(
        self, competitor_profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """
        Detect strategy patterns from competitors.

        Args:
            competitor_profiles: List of competitor profiles

        Returns:
            Detected strategies and opportunities
        """
        if not competitor_profiles:
            return {"strategies": [], "opportunities": []}

        strategies = []
        opportunities = []

        # Analyze posting frequency patterns
        freq_pattern = self._analyze_frequency_pattern(competitor_profiles)
        if freq_pattern:
            strategies.append(freq_pattern)

        # Analyze content type distribution
        content_pattern = self._analyze_content_types(competitor_profiles)
        if content_pattern:
            strategies.append(content_pattern)

        # Analyze topic focus
        topic_pattern = self._analyze_topic_focus(competitor_profiles)
        if topic_pattern:
            strategies.append(topic_pattern)

        # Identify opportunities from patterns
        if freq_pattern:
            opportunities.append(
                f"Consider {freq_pattern.get('recommendation', 'adjusting posting frequency')}"
            )

        if content_pattern:
            opportunities.append(
                f"Explore {content_pattern.get('underutilized_types', ['different content types'])[0]}"
            )

        return {"strategies": strategies, "opportunities": opportunities}

    def _analyze_frequency_pattern(
        self, profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """Analyze posting frequency patterns."""
        frequencies = [p.posts_per_week for p in profiles]
        avg_frequency = sum(frequencies) / len(frequencies)

        if avg_frequency >= 5:
            pattern = "high_frequency"
            recommendation = "posting 5+ times per week for visibility"
        elif avg_frequency >= 3:
            pattern = "moderate_frequency"
            recommendation = "posting 3-5 times per week for consistency"
        else:
            pattern = "low_frequency"
            recommendation = "focusing on quality over quantity"

        return {
            "pattern": pattern,
            "avg_frequency": round(avg_frequency, 1),
            "recommendation": recommendation,
        }

    def _analyze_content_types(
        self, profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """Analyze content type distribution."""
        all_types: Dict[str, int] = {}

        for profile in profiles:
            for content_type, count in profile.content_types.items():
                all_types[content_type] = all_types.get(content_type, 0) + count

        if not all_types:
            return {}

        # Find most common types
        total = sum(all_types.values())
        type_percentages = {t: (c / total) for t, c in all_types.items()}

        most_common = max(type_percentages.items(), key=lambda x: x[1])

        # Find underutilized types
        all_possible_types = {"article", "tutorial", "opinion", "news", "case_study"}
        underutilized = list(all_possible_types - set(all_types.keys()))

        return {
            "most_common_type": most_common[0],
            "most_common_percentage": round(most_common[1] * 100, 1),
            "underutilized_types": underutilized,
        }

    def _analyze_topic_focus(
        self, profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """Analyze topic focus patterns."""
        all_topics: List[str] = []

        for profile in profiles:
            topics = [topic for topic, _ in profile.top_topics]
            all_topics.extend(topics)

        if not all_topics:
            return {}

        # Count topic frequency
        topic_counts = Counter(all_topics)
        most_common_topics = topic_counts.most_common(5)

        return {
            "trending_topics": [topic for topic, _ in most_common_topics],
            "topic_concentration": len(set(all_topics)) / len(all_topics)
            if all_topics
            else 0,
        }