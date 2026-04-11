"""Temporal feature extraction."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.base import BaseFeatureExtractor

logger = get_logger(__name__)

# Platform-specific optimal hours (in UTC, adjust for timezone if needed)
PLATFORM_PEAK_HOURS = {
    "linkedin": [8, 9, 10, 12, 13, 17, 18],  # 8-10 AM, 12-1 PM, 5-6 PM
    "twitter": [12, 13, 17, 18, 9, 10],  # 12-1 PM, 5-6 PM, weekends 9-10 AM
    "bluesky": [12, 13, 17, 18, 9, 10],  # Similar to Twitter
}


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract time-based features from posts."""

    @property
    def feature_names(self) -> List[str]:
        """Return list of temporal feature names."""
        return [
            "hour",
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_business_hours",
            "is_morning",
            "is_afternoon",
            "is_evening",
            "is_night",
            "is_peak_hour",
            "time_since_midnight",
            "time_until_midnight",
            "days_since_last_post",
            "hours_since_last_post",
            "posts_in_last_24h",
            "posts_in_last_7d",
            "avg_posting_interval_hours",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from DataFrame.

        Args:
            df: DataFrame with 'published_at' column (and optionally 'platform')

        Returns:
            DataFrame with temporal features

        Example:
            >>> extractor = TemporalFeatureExtractor()
            >>> features = extractor.extract(df)
            >>> print(features[['hour', 'day_of_week', 'is_weekend']].head())
        """
        self.validate_input(df, ["published_at"])

        result = pd.DataFrame(index=df.index)

        # Convert to datetime if needed
        published_at = pd.to_datetime(df["published_at"])

        # Basic time features
        result["hour"] = published_at.dt.hour
        result["day_of_week"] = published_at.dt.dayofweek  # 0=Monday, 6=Sunday
        result["day_of_month"] = published_at.dt.day
        result["week_of_year"] = published_at.dt.isocalendar().week
        result["month"] = published_at.dt.month
        result["quarter"] = published_at.dt.quarter
        result["year"] = published_at.dt.year

        # Weekend indicator
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

        # Time of day indicators
        result["is_business_hours"] = (
            (result["hour"] >= 9) & (result["hour"] <= 17)
        ).astype(int)
        result["is_morning"] = (
            (result["hour"] >= 6) & (result["hour"] < 12)
        ).astype(int)
        result["is_afternoon"] = (
            (result["hour"] >= 12) & (result["hour"] < 17)
        ).astype(int)
        result["is_evening"] = (
            (result["hour"] >= 17) & (result["hour"] < 22)
        ).astype(int)
        result["is_night"] = (
            (result["hour"] >= 22) | (result["hour"] < 6)
        ).astype(int)

        # Peak hour indicator (platform-specific if platform column exists)
        if "platform" in df.columns:
            result["is_peak_hour"] = df.apply(
                lambda row: int(
                    row["hour"] in PLATFORM_PEAK_HOURS.get(row["platform"], [])
                )
                if pd.notna(row["hour"])
                else 0,
                axis=1,
            )
        else:
            # Generic peak hours if no platform specified
            result["is_peak_hour"] = result["hour"].apply(
                lambda h: int(h in [9, 12, 17])
            )

        # Time since/until midnight (in minutes)
        result["time_since_midnight"] = result["hour"] * 60 + published_at.dt.minute
        result["time_until_midnight"] = 1440 - result["time_since_midnight"]

        # Recency features (require sorting by published_at)
        df_sorted = df.copy()
        df_sorted["published_at"] = published_at
        df_sorted = df_sorted.sort_values("published_at")

        # Days/hours since last post
        time_diff = df_sorted["published_at"].diff()
        result["days_since_last_post"] = time_diff.dt.total_seconds() / 86400
        result["hours_since_last_post"] = time_diff.dt.total_seconds() / 3600

        # Fill first post (no previous post)
        result["days_since_last_post"] = result["days_since_last_post"].fillna(0)
        result["hours_since_last_post"] = result["hours_since_last_post"].fillna(0)

        # Posts in last N days
        result["posts_in_last_24h"] = 0
        result["posts_in_last_7d"] = 0

        for idx in df_sorted.index:
            current_time = df_sorted.loc[idx, "published_at"]
            result.loc[idx, "posts_in_last_24h"] = int(
                (
                    (df_sorted["published_at"] >= current_time - pd.Timedelta(days=1))
                    & (df_sorted["published_at"] < current_time)
                ).sum()
            )
            result.loc[idx, "posts_in_last_7d"] = int(
                (
                    (df_sorted["published_at"] >= current_time - pd.Timedelta(days=7))
                    & (df_sorted["published_at"] < current_time)
                ).sum()
            )

        # Average posting interval
        result["avg_posting_interval_hours"] = result["hours_since_last_post"].expanding().mean()

        logger.info(f"Extracted {len(result.columns)} temporal features")

        return result

    def extract_single(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract temporal features from single post.

        Args:
            post_data: Dictionary with 'published_at' key

        Returns:
            Dictionary with temporal features
        """
        published_at_str = post_data.get("published_at")
        platform = post_data.get("platform", "")

        if not published_at_str:
            # Return zero features if no published_at
            return {name: 0 for name in self.feature_names}

        # Parse datetime
        if isinstance(published_at_str, str):
            published_at = pd.to_datetime(published_at_str)
        else:
            published_at = published_at_str

        hour = published_at.hour
        day_of_week = published_at.dayofweek

        features = {
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": published_at.day,
            "week_of_year": published_at.isocalendar()[1],
            "month": published_at.month,
            "quarter": (published_at.month - 1) // 3 + 1,
            "year": published_at.year,
            "is_weekend": int(day_of_week >= 5),
            "is_business_hours": int(9 <= hour <= 17),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 17),
            "is_evening": int(17 <= hour < 22),
            "is_night": int(hour >= 22 or hour < 6),
            "is_peak_hour": int(
                hour in PLATFORM_PEAK_HOURS.get(platform, [9, 12, 17])
            ),
            "time_since_midnight": hour * 60 + published_at.minute,
            "time_until_midnight": 1440 - (hour * 60 + published_at.minute),
            # Recency features require historical data, set to 0 for single post
            "days_since_last_post": 0,
            "hours_since_last_post": 0,
            "posts_in_last_24h": 0,
            "posts_in_last_7d": 0,
            "avg_posting_interval_hours": 0,
        }

        return features