"""Interest mapping from audience engagement."""

from typing import Any, Dict, List

from collections import Counter

from bufferiq.ml.segmentation.types import AudienceDataPoint


class InterestMapper:
    """Map interests and topics from audience data."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize interest mapper."""
        self.config = config or {}
        self.max_topics = self.config.get("max_topics", 10)

    def map(
        self, audience_members: List[AudienceDataPoint], platform: str
    ) -> Dict[str, List[str]]:
        """
        Map interests from audience members.

        Args:
            audience_members: List of audience members
            platform: Platform type

        Returns:
            Dictionary with primary, secondary, and avoided topics
        """
        if not audience_members:
            return self._empty_interests()

        all_topics = []
        for member in audience_members:
            all_topics.extend(member.topics_engaged)

        if not all_topics:
            return self._empty_interests()

        # Count topic frequencies
        topic_counts = Counter(all_topics)
        sorted_topics = [topic for topic, _ in topic_counts.most_common()]

        # Split into primary and secondary
        primary_count = max(1, len(sorted_topics) // 2)
        primary_topics = sorted_topics[:primary_count]
        secondary_topics = sorted_topics[primary_count : primary_count * 2]

        # Avoided topics are those with very low engagement
        avoided_topics = self._infer_avoided_topics(audience_members, sorted_topics)

        return {
            "primary": primary_topics[: self.max_topics],
            "secondary": secondary_topics[: self.max_topics],
            "avoided": avoided_topics[: self.max_topics],
        }

    def _infer_avoided_topics(
        self, audience_members: List[AudienceDataPoint], engaged_topics: List[str]
    ) -> List[str]:
        """Infer topics avoided by the segment."""
        # Common platform-specific topics not in engaged list
        platform_common_topics = [
            "politics",
            "technology",
            "sports",
            "entertainment",
            "business",
            "health",
            "finance",
            "science",
        ]

        engaged_set = set(engaged_topics)
        avoided = [t for t in platform_common_topics if t not in engaged_set]

        return avoided

    def _empty_interests(self) -> Dict[str, List[str]]:
        """Return empty interests."""
        return {
            "primary": ["general"],
            "secondary": [],
            "avoided": [],
        }