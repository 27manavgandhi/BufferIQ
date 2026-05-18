"""Topic overlap analysis."""

from typing import Any, Dict, List
import logging

from bufferiq.ml.gaps.competitors.analyzer import CompetitorProfile

logger = logging.getLogger(__name__)


class OverlapAnalyzer:
    """
    Analyze topic overlap between user and competitors.

    Identifies unique topics, common topics, and gaps.
    """

    def analyze(
        self, user_profile: CompetitorProfile, competitor_profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """
        Analyze topic overlap.

        Args:
            user_profile: User's profile
            competitor_profiles: Competitor profiles

        Returns:
            Overlap analysis
        """
        # Extract topics
        user_topics = {topic for topic, _ in user_profile.top_topics}

        competitor_topics: set = set()
        for profile in competitor_profiles:
            topics = {topic for topic, _ in profile.top_topics}
            competitor_topics.update(topics)

        # Calculate overlaps
        unique_topics = list(user_topics - competitor_topics)
        missed_topics = list(competitor_topics - user_topics)
        common_topics = list(user_topics & competitor_topics)

        # Find gaps where NO competitors cover
        all_topics_covered = competitor_topics
        potential_opportunities = self._find_potential_opportunities(
            all_topics_covered
        )

        # Calculate overlap percentage
        if user_topics:
            overlap_percentage = (len(common_topics) / len(user_topics)) * 100
        else:
            overlap_percentage = 0.0

        return {
            "unique_topics": unique_topics,
            "missed_topics": missed_topics,
            "common_topics": common_topics,
            "competitor_gaps": potential_opportunities,
            "overlap_percentage": round(overlap_percentage, 2),
        }

    def _find_potential_opportunities(
        self, covered_topics: set
    ) -> List[str]:
        """Find topics not covered by anyone."""
        # All possible topics in industry
        all_industry_topics = {
            "AI & Machine Learning",
            "Cloud Computing",
            "Cybersecurity",
            "DevOps",
            "Data Science",
            "Blockchain",
            "IoT",
            "Edge Computing",
            "Quantum Computing",
            "5G Technology",
        }

        uncovered = all_industry_topics - covered_topics
        return list(uncovered)