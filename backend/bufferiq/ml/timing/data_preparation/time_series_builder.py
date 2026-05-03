"""Build time-series from post data."""

import pandas as pd
from typing import Optional, List
from datetime import datetime


class TimeSeriesBuilder:
    """Build time-series from post data for temporal analysis."""

    SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]

    def __init__(self) -> None:
        """Initialize TimeSeriesBuilder."""
        pass

    def build_hourly_series(
        self,
        posts: pd.DataFrame,
        platform: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build hourly time-series of engagement.

        Args:
            posts: Posts DataFrame with published_at, likes, comments, shares, engagement_score
            platform: Filter by platform (linkedin/twitter/bluesky)

        Returns:
            DataFrame with columns: timestamp, engagement_score, likes, comments, shares, post_count

        Example:
            >>> builder = TimeSeriesBuilder()
            >>> ts = builder.build_hourly_series(posts_df, platform='linkedin')
            >>> assert 'timestamp' in ts.columns
            >>> assert ts['timestamp'].is_monotonic_increasing
        """
        if posts.empty:
            return self._empty_time_series()

        # Validate platform
        if platform is not None:
            self._validate_platform(platform)
            posts = posts[posts["platform"] == platform].copy()

        if posts.empty:
            return self._empty_time_series()

        # Ensure published_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(posts["published_at"]):
            posts["published_at"] = pd.to_datetime(posts["published_at"])

        # Round to hour
        posts["hour"] = posts["published_at"].dt.floor("H")

        # Aggregate by hour
        aggregations = {
            "engagement_score": "mean",
            "likes": "sum",
            "comments": "sum",
            "shares": "sum",
            "id": "count",
        }

        # Only aggregate columns that exist
        available_aggs = {
            k: v for k, v in aggregations.items() if k in posts.columns or k == "id"
        }

        ts = posts.groupby("hour").agg(available_aggs)
        ts = ts.rename(columns={"id": "post_count"})

        # Reset index
        ts = ts.reset_index().rename(columns={"hour": "timestamp"})

        # Fill missing hours
        ts = self._fill_missing_hours(ts)

        return ts

    def build_daily_series(
        self,
        posts: pd.DataFrame,
        platform: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build daily time-series of engagement.

        Args:
            posts: Posts DataFrame
            platform: Filter by platform

        Returns:
            DataFrame with daily aggregation
        """
        if posts.empty:
            return self._empty_time_series()

        # Validate platform
        if platform is not None:
            self._validate_platform(platform)
            posts = posts[posts["platform"] == platform].copy()

        if posts.empty:
            return self._empty_time_series()

        # Ensure published_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(posts["published_at"]):
            posts["published_at"] = pd.to_datetime(posts["published_at"])

        # Round to day
        posts["day"] = posts["published_at"].dt.floor("D")

        # Aggregate by day
        aggregations = {
            "engagement_score": "mean",
            "likes": "sum",
            "comments": "sum",
            "shares": "sum",
            "id": "count",
        }

        available_aggs = {
            k: v for k, v in aggregations.items() if k in posts.columns or k == "id"
        }

        ts = posts.groupby("day").agg(available_aggs)
        ts = ts.rename(columns={"id": "post_count"})

        # Reset index
        ts = ts.reset_index().rename(columns={"day": "timestamp"})

        # Fill missing days
        ts = self._fill_missing_days(ts)

        return ts

    def build_by_platform(
        self,
        posts: pd.DataFrame,
        platform: str,
    ) -> pd.DataFrame:
        """
        Build hourly time-series for a specific platform.

        Args:
            posts: Posts DataFrame
            platform: Platform to filter (linkedin/twitter/bluesky)

        Returns:
            Hourly time-series for the platform
        """
        self._validate_platform(platform)
        return self.build_hourly_series(posts, platform=platform)

    def _validate_platform(self, platform: str) -> None:
        """Validate platform is supported."""
        if platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' is not supported. "
                f"Supported platforms: {self.SUPPORTED_PLATFORMS}"
            )

    def _fill_missing_hours(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Fill missing hours with zeros."""
        if ts.empty:
            return ts

        # Create full hour range
        full_range = pd.date_range(
            start=ts["timestamp"].min(),
            end=ts["timestamp"].max(),
            freq="H",
        )

        # Reindex and fill
        ts = ts.set_index("timestamp").reindex(full_range, fill_value=0.0)
        ts = ts.reset_index().rename(columns={"index": "timestamp"})

        return ts

    def _fill_missing_days(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Fill missing days with zeros."""
        if ts.empty:
            return ts

        # Create full day range
        full_range = pd.date_range(
            start=ts["timestamp"].min(),
            end=ts["timestamp"].max(),
            freq="D",
        )

        # Reindex and fill
        ts = ts.set_index("timestamp").reindex(full_range, fill_value=0.0)
        ts = ts.reset_index().rename(columns={"index": "timestamp"})

        return ts

    def _empty_time_series(self) -> pd.DataFrame:
        """Return empty time-series DataFrame with correct columns."""
        return pd.DataFrame(
            columns=[
                "timestamp",
                "engagement_score",
                "likes",
                "comments",
                "shares",
                "post_count",
            ]
        )