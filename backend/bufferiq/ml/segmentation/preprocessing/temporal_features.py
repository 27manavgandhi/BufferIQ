"""Temporal feature engineering."""

from typing import List

import numpy as np


class TemporalFeatureExtractor:
    """Extract temporal features from activity patterns."""

    def extract_hour_features(self, active_hours: List[int]) -> dict[str, float]:
        """
        Extract features from active hours.

        Args:
            active_hours: List of active hours (0-23)

        Returns:
            Dictionary of temporal features
        """
        if not active_hours:
            return {
                "active_hour_spread": 0.0,
                "peak_activity_hour": 12.0,
                "morning_active": 0.0,
                "afternoon_active": 0.0,
                "evening_active": 0.0,
            }

        unique_hours = list(set(active_hours))
        return {
            "active_hour_spread": len(unique_hours) / 24.0,
            "peak_activity_hour": float(np.mean(active_hours)),
            "morning_active": float(any(h in range(6, 12) for h in unique_hours)),
            "afternoon_active": float(any(h in range(12, 18) for h in unique_hours)),
            "evening_active": float(any(h in range(18, 24) or h in range(0, 6) for h in unique_hours)),
        }

    def extract_day_features(self, active_days: List[int]) -> dict[str, float]:
        """
        Extract features from active days.

        Args:
            active_days: List of active days (0-6, 0=Monday)

        Returns:
            Dictionary of day-based features
        """
        if not active_days:
            return {
                "active_day_spread": 0.0,
                "weekday_active": 0.0,
                "weekend_active": 0.0,
            }

        unique_days = list(set(active_days))
        return {
            "active_day_spread": len(unique_days) / 7.0,
            "weekday_active": float(
                any(d in range(0, 5) for d in unique_days)
            ),
            "weekend_active": float(any(d in range(5, 7) for d in unique_days)),
        }

    def combine_temporal_features(
        self, active_hours: List[int], active_days: List[int]
    ) -> dict[str, float]:
        """
        Combine hour and day features.

        Args:
            active_hours: List of active hours
            active_days: List of active days

        Returns:
            Combined temporal features
        """
        hour_features = self.extract_hour_features(active_hours)
        day_features = self.extract_day_features(active_days)

        return {**hour_features, **day_features}