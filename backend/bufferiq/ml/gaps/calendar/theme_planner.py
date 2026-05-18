"""Thematic content planning."""

from typing import Any, Dict, List
import logging
from collections import Counter

from bufferiq.ml.gaps.recommendations.generator import ContentRecommendation

logger = logging.getLogger(__name__)


class ThemePlanner:
    """
    Plan thematic weeks for content calendar.

    Groups related content into themed weeks for coherence.
    """

    def plan_themes(
        self,
        recommendations: List[ContentRecommendation],
        weeks: int,
    ) -> List[Dict[str, Any]]:
        """
        Plan theme weeks.

        Args:
            recommendations: Content recommendations
            weeks: Number of weeks

        Returns:
            List of theme week definitions
        """
        # Group recommendations by topic similarity
        topic_groups = self._group_by_topic(recommendations)

        # Assign themes to weeks
        theme_weeks = []

        for week_num in range(1, weeks + 1):
            # Pick most relevant theme for this week
            if topic_groups:
                theme_name, theme_topics = topic_groups.pop(0)
            else:
                theme_name = f"Week {week_num} Focus"
                theme_topics = []

            theme_weeks.append({
                "week": week_num,
                "theme": theme_name,
                "topics": theme_topics,
                "description": f"Focus on {theme_name.lower()} topics",
            })

        return theme_weeks

    def _group_by_topic(
        self, recommendations: List[ContentRecommendation]
    ) -> List[tuple]:
        """Group recommendations by topic similarity."""
        # Extract main topic categories
        topics = [rec.topic for rec in recommendations]

        # Count topic frequency
        topic_counts = Counter(topics)

        # Group similar topics
        groups = []

        # Get unique topics sorted by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        for topic, count in sorted_topics:
            if count >= 2:
                # Get related recommendations
                related = [rec for rec in recommendations if rec.topic == topic]
                groups.append((topic, [r.topic for r in related[:3]]))

        return groups

    def suggest_theme_name(self, topics: List[str]) -> str:
        """
        Suggest a theme name for grouped topics.

        Args:
            topics: List of topics

        Returns:
            Theme name
        """
        if not topics:
            return "General Content"

        # Extract common keywords
        all_keywords = []
        for topic in topics:
            words = topic.split()
            all_keywords.extend(words)

        # Find most common keyword
        keyword_counts = Counter(all_keywords)
        if keyword_counts:
            most_common = keyword_counts.most_common(1)[0][0]
            return f"{most_common.title()} Week"

        return topics[0]