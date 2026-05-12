"""
Temporal diversity analysis.

Measures diversity of posting times and patterns.
"""

from collections import Counter
from datetime import datetime
from typing import List

import numpy as np


class TemporalDiversityAnalyzer:
    """
        Analyze temporal diversity.

        Measures how diverse posting times are.

        Example:
    ```python
            analyzer = TemporalDiversityAnalyzer()
            times = [
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 2, 14, 0),
            ]
            diversity = analyzer.calculate_diversity(times)
            print(f"Diversity: {diversity:.2f}")
    ```
    """

    def __init__(self) -> None:
        """Initialize temporal diversity analyzer."""
        pass

    def calculate_diversity(self, timestamps: List[datetime]) -> float:
        """
        Calculate temporal diversity.

        Args:
            timestamps: List of posting timestamps

        Returns:
            Diversity score (0-1), higher = more diverse

        Raises:
            ValueError: If timestamps list is empty
        """
        if not timestamps:
            raise ValueError("Timestamps list cannot be empty")

        if len(timestamps) < 2:
            return 0.0

        # Extract hours of day
        hours = [ts.hour for ts in timestamps]

        # Calculate hour distribution entropy
        hour_counts = Counter(hours)
        total = len(hours)

        entropy = 0.0
        for count in hour_counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability)

        # Normalize (max entropy for 24 hours)
        max_entropy = np.log2(24)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def calculate_weekday_diversity(self, timestamps: List[datetime]) -> float:
        """
        Calculate weekday diversity.

        Args:
            timestamps: List of posting timestamps

        Returns:
            Diversity score (0-1)

        Raises:
            ValueError: If timestamps list is empty
        """
        if not timestamps:
            raise ValueError("Timestamps list cannot be empty")

        if len(timestamps) < 2:
            return 0.0

        # Extract weekdays (0=Monday, 6=Sunday)
        weekdays = [ts.weekday() for ts in timestamps]

        # Calculate weekday distribution entropy
        weekday_counts = Counter(weekdays)
        total = len(weekdays)

        entropy = 0.0
        for count in weekday_counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability)

        # Normalize (max entropy for 7 days)
        max_entropy = np.log2(7)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def calculate_time_intervals(self, timestamps: List[datetime]) -> List[float]:
        """
        Calculate time intervals between posts.

        Args:
            timestamps: List of posting timestamps (sorted)

        Returns:
            List of intervals in hours

        Raises:
            ValueError: If timestamps list is empty
        """
        if not timestamps:
            raise ValueError("Timestamps list cannot be empty")

        if len(timestamps) < 2:
            return []

        # Sort timestamps
        sorted_times = sorted(timestamps)

        # Calculate intervals
        intervals = []
        for i in range(1, len(sorted_times)):
            delta = sorted_times[i] - sorted_times[i - 1]
            hours = delta.total_seconds() / 3600
            intervals.append(hours)

        return intervals
