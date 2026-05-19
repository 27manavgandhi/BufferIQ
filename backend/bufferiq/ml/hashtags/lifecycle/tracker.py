"""
Hashtag lifecycle tracker.

Tracks hashtag through lifecycle stages.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class LifecycleStage:
    """Hashtag lifecycle stage."""

    stage: str  # "nascent", "growing", "mature", "declining", "dormant"
    entered_at: datetime
    duration: Optional[timedelta] = None
    volume_at_entry: int = 0


@dataclass
class HashtagLifecycle:
    """Complete hashtag lifecycle."""

    hashtag: str
    current_stage: str
    age_days: int
    stages_history: List[LifecycleStage]
    predicted_lifetime: Optional[int] = None  # Days


class LifecycleTracker:
    """
    Track hashtag lifecycle.

    Example:
```python
        tracker = LifecycleTracker()

        lifecycle = tracker.track(
            hashtag="ai",
            volume_history=[50, 100, 200, 500, 1000, 900, 700],
            start_date=datetime.now() - timedelta(days=180)
        )

        print(f"Hashtag: #{lifecycle.hashtag}")
        print(f"  Current stage: {lifecycle.current_stage}")
        print(f"  Age: {lifecycle.age_days} days")
```
    """

    def track(
        self,
        hashtag: str,
        volume_history: List[int],
        start_date: datetime,
    ) -> HashtagLifecycle:
        """
        Track hashtag lifecycle.

        Args:
            hashtag: Hashtag to track
            volume_history: Volume over time
            start_date: When tracking started

        Returns:
            Lifecycle information
        """
        # Determine current stage
        current_stage = self._determine_stage(volume_history)

        # Calculate age
        age_days = (datetime.now() - start_date).days

        # Build stage history (simplified)
        stages_history = [
            LifecycleStage(
                stage=current_stage,
                entered_at=datetime.now(),
                volume_at_entry=volume_history[-1] if volume_history else 0,
            )
        ]

        # Predict lifetime
        predicted_lifetime = self._predict_lifetime(volume_history, age_days)

        return HashtagLifecycle(
            hashtag=hashtag,
            current_stage=current_stage,
            age_days=age_days,
            stages_history=stages_history,
            predicted_lifetime=predicted_lifetime,
        )

    def _determine_stage(self, volume_history: List[int]) -> str:
        """Determine current lifecycle stage."""
        if not volume_history:
            return "nascent"

        if len(volume_history) < 3:
            return "nascent"

        # Calculate trend
        recent = volume_history[-3:]
        if recent[-1] > recent[0] * 1.5:
            return "growing"
        elif recent[-1] < recent[0] * 0.7:
            return "declining"
        elif recent[-1] < 10:
            return "dormant"
        else:
            return "mature"

    def _predict_lifetime(
        self, volume_history: List[int], current_age: int
    ) -> int | None:
        """Predict total lifetime in days."""
        if not volume_history:
            return None

        # Simple prediction: if declining, estimate based on rate
        if len(volume_history) >= 3:
            trend = volume_history[-1] - volume_history[-3]
            if trend < 0:
                # Declining - estimate days until dormant
                current_vol = volume_history[-1]
                decline_rate = abs(trend) / 2  # Per day
                if decline_rate > 0:
                    days_remaining = int(current_vol / decline_rate)
                    return current_age + days_remaining

        # Default estimate
        return current_age + 90