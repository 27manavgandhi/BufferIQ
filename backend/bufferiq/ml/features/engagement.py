"""Engagement feature extraction."""

from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Post
from bufferiq.ml.features.base import BaseFeatureExtractor

logger = get_logger(__name__)


class EngagementFeatureExtractor(BaseFeatureExtractor):
    """Extract engagement-based features."""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        """
        Initialize engagement feature extractor.

        Args:
            session: Optional AsyncSession for database queries
        """
        self.session = session

    @property
    def feature_names(self) -> List[str]:
        """Return list of engagement feature names."""
        return [
            "user_avg_likes",
            "user_avg_comments",
            "user_avg_shares",
            "user_avg_engagement_rate",
            "user_median_engagement_rate",
            "user_post_count",
            "platform_avg_likes",
            "platform_avg_comments",
            "platform_avg_shares",
            "platform_avg_engagement_rate",
            "engagement_rate_last_5",
            "engagement_rate_last_10",
            "engagement_trend",
            "is_improving",
            "best_post_engagement",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract engagement features from DataFrame.

        Note: This is a simplified version that works with DataFrame only.
        For full database integration, use extract_async().

        Args:
            df: DataFrame with engagement columns

        Returns:
            DataFrame with engagement features
        """
        result = pd.DataFrame(index=df.index)

        # Required columns
        required_cols = ["likes", "comments", "shares", "impressions"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns for engagement features: {missing_cols}")
            # Return zero features
            for feature_name in self.feature_names:
                result[feature_name] = 0
            return result

        # Calculate engagement rate
        df["engagement_rate"] = (
            (df["likes"] + df["comments"] + df["shares"])
            / df["impressions"].replace(0, 1)
            * 100
        )

        # User-level features (average across all posts for this user)
        if "user_id" in df.columns:
            user_stats = df.groupby("user_id").agg(
                {
                    "likes": "mean",
                    "comments": "mean",
                    "shares": "mean",
                    "engagement_rate": ["mean", "median", "count"],
                }
            )

            result["user_avg_likes"] = df["user_id"].map(
                user_stats["likes"]["mean"]
            ).fillna(0)
            result["user_avg_comments"] = df["user_id"].map(
                user_stats["comments"]["mean"]
            ).fillna(0)
            result["user_avg_shares"] = df["user_id"].map(
                user_stats["shares"]["mean"]
            ).fillna(0)
            result["user_avg_engagement_rate"] = df["user_id"].map(
                user_stats["engagement_rate"]["mean"]
            ).fillna(0)
            result["user_median_engagement_rate"] = df["user_id"].map(
                user_stats["engagement_rate"]["median"]
            ).fillna(0)
            result["user_post_count"] = df["user_id"].map(
                user_stats["engagement_rate"]["count"]
            ).fillna(0)
        else:
            result["user_avg_likes"] = df["likes"].mean()
            result["user_avg_comments"] = df["comments"].mean()
            result["user_avg_shares"] = df["shares"].mean()
            result["user_avg_engagement_rate"] = df["engagement_rate"].mean()
            result["user_median_engagement_rate"] = df["engagement_rate"].median()
            result["user_post_count"] = len(df)

        # Platform-level features
        if "platform" in df.columns:
            platform_stats = df.groupby("platform").agg(
                {
                    "likes": "mean",
                    "comments": "mean",
                    "shares": "mean",
                    "engagement_rate": "mean",
                }
            )

            result["platform_avg_likes"] = df["platform"].map(
                platform_stats["likes"]
            ).fillna(0)
            result["platform_avg_comments"] = df["platform"].map(
                platform_stats["comments"]
            ).fillna(0)
            result["platform_avg_shares"] = df["platform"].map(
                platform_stats["shares"]
            ).fillna(0)
            result["platform_avg_engagement_rate"] = df["platform"].map(
                platform_stats["engagement_rate"]
            ).fillna(0)
        else:
            result["platform_avg_likes"] = df["likes"].mean()
            result["platform_avg_comments"] = df["comments"].mean()
            result["platform_avg_shares"] = df["shares"].mean()
            result["platform_avg_engagement_rate"] = df["engagement_rate"].mean()

        # Rolling window features (last N posts)
        result["engagement_rate_last_5"] = (
            df["engagement_rate"].rolling(window=5, min_periods=1).mean()
        )
        result["engagement_rate_last_10"] = (
            df["engagement_rate"].rolling(window=10, min_periods=1).mean()
        )

        # Engagement trend (slope of last 10 posts)
        def calculate_trend(series: pd.Series) -> float:
            if len(series) < 2:
                return 0.0
            x = range(len(series))
            y = series.values
            # Simple linear regression slope
            n = len(x)
            slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (
                n * sum(xi**2 for xi in x) - sum(x) ** 2
            )
            return slope

        result["engagement_trend"] = (
            df["engagement_rate"]
            .rolling(window=10, min_periods=2)
            .apply(calculate_trend, raw=False)
            .fillna(0)
        )

        result["is_improving"] = (result["engagement_trend"] > 0).astype(int)

        # Best post engagement in last 30 days
        if "published_at" in df.columns:
            df_copy = df.copy()
            df_copy["published_at"] = pd.to_datetime(df_copy["published_at"])

            def best_in_last_30d(row: pd.Series) -> float:
                cutoff = row["published_at"] - pd.Timedelta(days=30)
                recent = df_copy[
                    (df_copy["published_at"] >= cutoff)
                    & (df_copy["published_at"] < row["published_at"])
                ]
                return recent["engagement_rate"].max() if len(recent) > 0 else 0.0

            result["best_post_engagement"] = df_copy.apply(
                best_in_last_30d, axis=1
            ).fillna(0)
        else:
            result["best_post_engagement"] = 0

        logger.info(f"Extracted {len(result.columns)} engagement features")

        return result

    def extract_single(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract engagement features from single post.

        Note: Single post extraction returns zeros for historical features.

        Args:
            post_data: Dictionary with post data

        Returns:
            Dictionary with engagement features
        """
        # For single post, we can't compute historical features
        # Return zeros as placeholders
        return {name: 0 for name in self.feature_names}

    async def extract_async(
        self, df: pd.DataFrame, session: AsyncSession
    ) -> pd.DataFrame:
        """
        Extract engagement features with database queries.

        Args:
            df: DataFrame with post data
            session: AsyncSession for database queries

        Returns:
            DataFrame with engagement features
        """
        # For now, use the simpler DataFrame-based extraction
        # In production, you would query the database for historical stats
        return self.extract(df)