"""
Real-time hashtag trend monitor.

Monitors trending hashtags in real-time.
"""

from datetime import datetime, timedelta
from typing import Dict, List
from collections import deque


class RealtimeMonitor:
    """
    Monitor hashtag trends in real-time.

    Example:
```python
        monitor = RealtimeMonitor(window_minutes=60)

        # Record usage
        monitor.record_usage("ai", datetime.now())

        # Get current rate
        rate = monitor.get_usage_rate("ai")
        print(f"Usage rate: {rate:.1f} per hour")
```
    """

    def __init__(self, window_minutes: int = 60) -> None:
        """
        Initialize realtime monitor.

        Args:
            window_minutes: Time window for rate calculation
        """
        self.window_minutes = window_minutes
        self.usage_log: Dict[str, deque] = {}

    def record_usage(self, hashtag: str, timestamp: datetime) -> None:
        """
        Record hashtag usage.

        Args:
            hashtag: Hashtag used
            timestamp: When it was used
        """
        if hashtag not in self.usage_log:
            self.usage_log[hashtag] = deque()

        self.usage_log[hashtag].append(timestamp)

        # Clean old entries
        self._clean_old_entries(hashtag)

    def get_usage_rate(self, hashtag: str) -> float:
        """
        Get current usage rate (uses per hour).

        Args:
            hashtag: Hashtag to check

        Returns:
            Usage rate per hour
        """
        if hashtag not in self.usage_log:
            return 0.0

        count = len(self.usage_log[hashtag])

        # Convert to hourly rate
        rate = (count / self.window_minutes) * 60

        return rate

    def get_trending(self, min_rate: float = 10.0) -> List[str]:
        """
        Get currently trending hashtags.

        Args:
            min_rate: Minimum rate to be considered trending

        Returns:
            List of trending hashtags
        """
        trending = []

        for hashtag in self.usage_log:
            rate = self.get_usage_rate(hashtag)
            if rate >= min_rate:
                trending.append(hashtag)

        # Sort by rate
        trending.sort(key=lambda h: self.get_usage_rate(h), reverse=True)

        return trending

    def _clean_old_entries(self, hashtag: str) -> None:
        """Remove entries outside time window."""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)

        while (
            self.usage_log[hashtag]
            and self.usage_log[hashtag][0] < cutoff
        ):
            self.usage_log[hashtag].popleft()