"""Behavioral profiling from audience data."""

from typing import Any, Dict, List

import numpy as np

from bufferiq.ml.segmentation.types import AudienceDataPoint


class BehavioralProfiler:
    """Profile behavioral characteristics of audience segments."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize behavioral profiler."""
        self.config = config or {}

    def profile(
        self,
        audience_members: List[AudienceDataPoint],
        feature_matrix: np.ndarray,
        platform: str,
    ) -> Dict[str, Any]:
        """
        Profile behavioral characteristics.

        Args:
            audience_members: List of audience members
            feature_matrix: Feature vectors
            platform: Platform type

        Returns:
            Dictionary of behavioral characteristics
        """
        if not audience_members:
            return self._empty_profile()

        primary_interaction = self._determine_primary_interaction(audience_members)
        content_preferences = self._calculate_content_preferences(audience_members)
        peak_hours = self._determine_peak_hours(audience_members)
        peak_days = self._determine_peak_days(audience_members)
        avg_engagement = self._calculate_avg_engagement(audience_members)
        avg_session_length = self._estimate_session_length(audience_members)
        posting_frequency = self._estimate_posting_frequency(audience_members)
        response_time = self._estimate_response_time(audience_members)

        return {
            "primary_interaction": primary_interaction,
            "content_preferences": content_preferences,
            "peak_hours": peak_hours,
            "peak_days": peak_days,
            "avg_engagement": avg_engagement,
            "avg_session_length": avg_session_length,
            "posting_frequency": posting_frequency,
            "response_time": response_time,
        }

    def _determine_primary_interaction(
        self, audience_members: List[AudienceDataPoint]
    ) -> str:
        """Determine primary interaction type."""
        interaction_totals: Dict[str, int] = {}

        for member in audience_members:
            for interaction_type, count in member.interaction_types.items():
                interaction_totals[interaction_type] = (
                    interaction_totals.get(interaction_type, 0) + count
                )

        if not interaction_totals:
            return "likes"

        return max(interaction_totals, key=interaction_totals.get)

    def _calculate_content_preferences(
        self, audience_members: List[AudienceDataPoint]
    ) -> Dict[str, float]:
        """Calculate content type preferences."""
        content_types = ["text", "image", "video", "link"]
        preferences: Dict[str, float] = {}

        for content_type in content_types:
            count = sum(
                1 for m in audience_members if content_type in m.content_types_engaged
            )
            preferences[content_type] = count / len(audience_members)

        return preferences

    def _determine_peak_hours(
        self, audience_members: List[AudienceDataPoint]
    ) -> List[int]:
        """Determine peak activity hours."""
        all_hours = []
        for member in audience_members:
            all_hours.extend(member.active_hours)

        if not all_hours:
            return [9, 12, 18]

        # Return hours with highest frequency
        from collections import Counter
        hour_counts = Counter(all_hours)
        peak_hours = [h for h, _ in hour_counts.most_common(3)]

        return sorted(peak_hours)

    def _determine_peak_days(
        self, audience_members: List[AudienceDataPoint]
    ) -> List[str]:
        """Determine peak activity days."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        all_days = []

        for member in audience_members:
            all_days.extend(member.active_days)

        if not all_days:
            return ["Monday", "Wednesday", "Friday"]

        from collections import Counter
        day_counts = Counter(all_days)
        peak_days = [day_names[d] for d, _ in day_counts.most_common(3)]

        return peak_days

    def _calculate_avg_engagement(
        self, audience_members: List[AudienceDataPoint]
    ) -> float:
        """Calculate average engagement rate."""
        if not audience_members:
            return 0.0

        total_engagement = sum(m.avg_engagement_rate for m in audience_members)
        return total_engagement / len(audience_members)

    def _estimate_session_length(
        self, audience_members: List[AudienceDataPoint]
    ) -> float:
        """Estimate average session length in minutes."""
        if not audience_members:
            return 30.0

        # Estimate from engagement and post count
        total_posts = sum(m.post_count for m in audience_members)
        avg_posts = total_posts / len(audience_members) if audience_members else 0

        # More posts typically correlates with longer sessions
        session_length = 20.0 + (avg_posts / 100.0 * 40.0)
        return min(max(session_length, 10.0), 120.0)

    def _estimate_posting_frequency(
        self, audience_members: List[AudienceDataPoint]
    ) -> str:
        """Estimate posting frequency."""
        if not audience_members:
            return "weekly"

        avg_posts = np.mean([m.post_count for m in audience_members])

        if avg_posts > 5:
            return "daily"
        elif avg_posts > 2:
            return "several_times_weekly"
        elif avg_posts > 1:
            return "weekly"
        else:
            return "occasionally"

    def _estimate_response_time(
        self, audience_members: List[AudienceDataPoint]
    ) -> float:
        """Estimate response time in hours."""
        if not audience_members:
            return 24.0

        # Higher engagement typically means faster response
        avg_engagement = np.mean([m.avg_engagement_rate for m in audience_members])
        response_hours = 24.0 * (1.0 - avg_engagement)

        return max(response_hours, 1.0)

    def _empty_profile(self) -> Dict[str, Any]:
        """Return empty profile."""
        return {
            "primary_interaction": "likes",
            "content_preferences": {
                "text": 0.25,
                "image": 0.25,
                "video": 0.25,
                "link": 0.25,
            },
            "peak_hours": [9, 12, 18],
            "peak_days": ["Monday", "Wednesday", "Friday"],
            "avg_engagement": 0.5,
            "avg_session_length": 30.0,
            "posting_frequency": "weekly",
            "response_time": 24.0,
        }