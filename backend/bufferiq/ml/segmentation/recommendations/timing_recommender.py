"""Timing recommendations for segments."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import PersonaProfile


class TimingRecommender:
    """Generate timing recommendations for personas."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize timing recommender."""
        self.config = config or {}

    def recommend(self, persona: PersonaProfile) -> Dict[str, Any]:
        """
        Generate timing recommendations.

        Args:
            persona: Persona profile

        Returns:
            Timing recommendations
        """
        posting_times = self._format_posting_times(persona.peak_activity_hours)
        posting_days = persona.peak_activity_days or ["Monday", "Wednesday", "Friday"]
        frequency = persona.posting_frequency_preference or "weekly"

        return {
            "optimal_posting_times": posting_times,
            "optimal_days": posting_days,
            "posting_frequency": frequency,
            "response_time_hours": persona.response_time_preference_hours,
        }

    def _format_posting_times(self, hours: List[int]) -> List[str]:
        """Format hours into time strings."""
        if not hours:
            hours = [9, 12, 18]

        return [f"{h:02d}:00" for h in sorted(hours)]