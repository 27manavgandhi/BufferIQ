"""Temporal pattern analysis."""

from typing import Any

import pandas as pd
from scipy import stats

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class TemporalAnalyzer:
    """Analyze time-based engagement patterns."""

    def hourly_patterns(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> pd.DataFrame:
        """
        Analyze engagement patterns by hour of day.

        Args:
            df: DataFrame with 'hour' and metric columns
            metric: Metric to analyze

        Returns:
            DataFrame with hourly statistics

        Example:
            >>> analyzer = TemporalAnalyzer()
            >>> hourly = analyzer.hourly_patterns(df, "engagement_rate")
            >>> print(hourly[['hour', 'mean', 'count']])
        """
        if "hour" not in df.columns:
            raise ValueError("DataFrame must have 'hour' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        # Group by hour
        hourly_stats = (
            df.groupby("hour")[metric]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )

        # Add confidence intervals
        hourly_stats["sem"] = hourly_stats["std"] / (hourly_stats["count"] ** 0.5)
        hourly_stats["ci_lower"] = hourly_stats["mean"] - 1.96 * hourly_stats["sem"]
        hourly_stats["ci_upper"] = hourly_stats["mean"] + 1.96 * hourly_stats["sem"]

        logger.info(
            "Analyzed hourly patterns",
            metric=metric,
            hours_with_data=len(hourly_stats),
            peak_hour=int(hourly_stats.loc[hourly_stats["mean"].idxmax(), "hour"]),
        )

        return hourly_stats

    def daily_patterns(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> pd.DataFrame:
        """
        Analyze engagement patterns by day of week.

        Args:
            df: DataFrame with 'day_of_week' and metric columns
            metric: Metric to analyze

        Returns:
            DataFrame with daily statistics

        Example:
            >>> daily = analyzer.daily_patterns(df, "engagement_rate")
            >>> print(daily[['day_name', 'mean', 'count']])
        """
        if "day_of_week" not in df.columns:
            raise ValueError("DataFrame must have 'day_of_week' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        # Group by day of week
        daily_stats = (
            df.groupby(["day_of_week", "day_name"])[metric]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )

        # Sort by day of week
        daily_stats = daily_stats.sort_values("day_of_week")

        # Add confidence intervals
        daily_stats["sem"] = daily_stats["std"] / (daily_stats["count"] ** 0.5)
        daily_stats["ci_lower"] = daily_stats["mean"] - 1.96 * daily_stats["sem"]
        daily_stats["ci_upper"] = daily_stats["mean"] + 1.96 * daily_stats["sem"]

        logger.info(
            "Analyzed daily patterns",
            metric=metric,
            days_with_data=len(daily_stats),
            best_day=daily_stats.loc[daily_stats["mean"].idxmax(), "day_name"],
        )

        return daily_stats

    def weekly_trends(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> pd.DataFrame:
        """
        Analyze weekly trends over time.

        Args:
            df: DataFrame with 'date' and metric columns
            metric: Metric to analyze

        Returns:
            DataFrame with weekly aggregated data

        Example:
            >>> weekly = analyzer.weekly_trends(df, "engagement_rate")
            >>> print(weekly[['week_start', 'mean', 'count']])
        """
        if "date" not in df.columns and "published_at" not in df.columns:
            raise ValueError("DataFrame must have 'date' or 'published_at' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Ensure datetime column
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        else:
            df["datetime"] = pd.to_datetime(df["published_at"])

        # Set as index for resampling
        df = df.set_index("datetime")

        # Resample to weekly frequency
        weekly_stats = df[metric].resample("W").agg(["mean", "median", "count"])
        weekly_stats = weekly_stats.reset_index()
        weekly_stats.columns = ["week_start", "mean", "median", "count"]

        # Calculate rolling average (4-week window)
        if len(weekly_stats) >= 4:
            weekly_stats["rolling_mean"] = (
                weekly_stats["mean"].rolling(window=4, min_periods=1).mean()
            )

        logger.info(
            "Analyzed weekly trends",
            metric=metric,
            weeks=len(weekly_stats),
            total_posts=int(weekly_stats["count"].sum()),
        )

        return weekly_stats

    def seasonal_patterns(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze seasonal patterns (monthly).

        Args:
            df: DataFrame with 'month' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with seasonal statistics

        Example:
            >>> seasonal = analyzer.seasonal_patterns(df, "engagement_rate")
            >>> print(seasonal['monthly_means'])
        """
        if "month" not in df.columns:
            raise ValueError("DataFrame must have 'month' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        # Group by month
        monthly_stats = (
            df.groupby("month")[metric]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )

        seasonal_info = {
            "monthly_means": monthly_stats.set_index("month")["mean"].to_dict(),
            "monthly_counts": monthly_stats.set_index("month")["count"].to_dict(),
            "peak_month": int(
                monthly_stats.loc[monthly_stats["mean"].idxmax(), "month"]
            ),
            "low_month": int(
                monthly_stats.loc[monthly_stats["mean"].idxmin(), "month"]
            ),
        }

        logger.info(
            "Analyzed seasonal patterns",
            metric=metric,
            months_with_data=len(monthly_stats),
            peak_month=seasonal_info["peak_month"],
        )

        return seasonal_info

    def optimal_posting_windows(
        self, df: pd.DataFrame, platform: str | None = None, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Identify optimal posting time windows.

        Args:
            df: DataFrame with temporal and engagement data
            platform: Filter by platform (optional)
            top_n: Number of top windows to return

        Returns:
            List of optimal time windows with statistics

        Example:
            >>> windows = analyzer.optimal_posting_windows(df, "linkedin", 5)
            >>> for w in windows:
            ...     print(f"{w['day_name']} {w['hour']}:00 - {w['mean_engagement']:.4f}")
        """
        required_cols = ["day_of_week", "day_name", "hour", "engagement_rate"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter by platform if specified
        if platform is not None:
            if "platform" not in df.columns:
                raise ValueError("DataFrame must have 'platform' column")
            df = df[df["platform"] == platform]

        if len(df) == 0:
            logger.warning("No data available for optimal window analysis")
            return []

        # Group by day and hour
        windows = (
            df.groupby(["day_of_week", "day_name", "hour"])["engagement_rate"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # Filter for minimum sample size (at least 3 posts)
        windows = windows[windows["count"] >= 3]

        if len(windows) == 0:
            logger.warning("No windows with sufficient sample size")
            return []

        # Sort by mean engagement
        windows = windows.sort_values("mean", ascending=False)

        # Get top N
        top_windows = windows.head(top_n)

        # Convert to list of dicts
        optimal_windows = [
            {
                "day_of_week": int(row["day_of_week"]),
                "day_name": row["day_name"],
                "hour": int(row["hour"]),
                "mean_engagement": float(row["mean"]),
                "post_count": int(row["count"]),
            }
            for _, row in top_windows.iterrows()
        ]

        logger.info(
            "Identified optimal posting windows",
            platform=platform or "all",
            top_n=top_n,
            windows_found=len(optimal_windows),
        )

        return optimal_windows

    def weekend_vs_weekday(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Compare weekend vs weekday performance.

        Args:
            df: DataFrame with 'is_weekend' and metric columns
            metric: Metric to compare

        Returns:
            Dictionary with comparison statistics

        Example:
            >>> comparison = analyzer.weekend_vs_weekday(df, "engagement_rate")
            >>> print(f"Weekend mean: {comparison['weekend_mean']:.4f}")
            >>> print(f"Weekday mean: {comparison['weekday_mean']:.4f}")
        """
        if "is_weekend" not in df.columns:
            raise ValueError("DataFrame must have 'is_weekend' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        weekend_data = df[df["is_weekend"]][metric].dropna()
        weekday_data = df[~df["is_weekend"]][metric].dropna()

        comparison = {
            "weekend_mean": float(weekend_data.mean())
            if len(weekend_data) > 0
            else 0.0,
            "weekday_mean": float(weekday_data.mean())
            if len(weekday_data) > 0
            else 0.0,
            "weekend_count": int(len(weekend_data)),
            "weekday_count": int(len(weekday_data)),
        }

        # Perform t-test if both have data
        if len(weekend_data) >= 2 and len(weekday_data) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(weekend_data, weekday_data)
                comparison["t_statistic"] = float(t_stat)
                comparison["p_value"] = float(p_value)
                comparison["significant"] = p_value < 0.05
            except Exception as e:
                logger.warning(f"Could not perform t-test: {e}")

        logger.info(
            "Compared weekend vs weekday",
            metric=metric,
            weekend_mean=comparison["weekend_mean"],
            weekday_mean=comparison["weekday_mean"],
            significant=comparison.get("significant", False),
        )

        return comparison
