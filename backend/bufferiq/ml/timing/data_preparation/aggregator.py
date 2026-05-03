"""Aggregate time-series data by different time windows."""

import pandas as pd
from typing import Dict, Any, List, Callable


class TemporalAggregator:
    """Aggregate engagement data by time windows."""

    def __init__(self) -> None:
        """Initialize TemporalAggregator."""
        pass

    def aggregate_hourly(
        self,
        ts: pd.DataFrame,
        agg_funcs: List[str] = ["mean", "median", "sum", "count", "std"],
    ) -> pd.DataFrame:
        """
        Aggregate time-series by hour of day (0-23).

        Args:
            ts: Time-series DataFrame
            agg_funcs: Aggregation functions to apply

        Returns:
            DataFrame aggregated by hour

        Example:
            >>> agg = TemporalAggregator()
            >>> hourly = agg.aggregate_hourly(ts)
            >>> assert len(hourly) <= 24
        """
        if ts.empty or "timestamp" not in ts.columns:
            return pd.DataFrame()

        # Extract hour
        df = ts.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        # Aggregate numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [col for col in numeric_cols if col != "hour"]

        if not numeric_cols:
            return pd.DataFrame()

        # Group and aggregate
        result = df.groupby("hour")[numeric_cols].agg(agg_funcs)

        # Flatten multi-index columns
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        result = result.reset_index()

        return result

    def aggregate_daily(
        self,
        ts: pd.DataFrame,
        agg_funcs: List[str] = ["mean", "median", "sum", "count", "std"],
    ) -> pd.DataFrame:
        """
        Aggregate time-series by day.

        Args:
            ts: Time-series DataFrame
            agg_funcs: Aggregation functions to apply

        Returns:
            DataFrame aggregated by day
        """
        if ts.empty or "timestamp" not in ts.columns:
            return pd.DataFrame()

        # Extract date
        df = ts.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # Aggregate numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [col for col in numeric_cols if col != "date"]

        if not numeric_cols:
            return pd.DataFrame()

        # Group and aggregate
        result = df.groupby("date")[numeric_cols].agg(agg_funcs)

        # Flatten multi-index columns
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        result = result.reset_index()

        return result

    def aggregate_by_dow(
        self,
        ts: pd.DataFrame,
        agg_funcs: List[str] = ["mean", "median", "sum", "count", "std"],
    ) -> pd.DataFrame:
        """
        Aggregate time-series by day of week (0=Monday, 6=Sunday).

        Args:
            ts: Time-series DataFrame
            agg_funcs: Aggregation functions to apply

        Returns:
            DataFrame aggregated by day of week

        Example:
            >>> agg = TemporalAggregator()
            >>> weekly = agg.aggregate_by_dow(ts)
            >>> assert len(weekly) <= 7
        """
        if ts.empty or "timestamp" not in ts.columns:
            return pd.DataFrame()

        # Extract day of week
        df = ts.copy()
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

        # Aggregate numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [col for col in numeric_cols if col != "day_of_week"]

        if not numeric_cols:
            return pd.DataFrame()

        # Group and aggregate
        result = df.groupby("day_of_week")[numeric_cols].agg(agg_funcs)

        # Flatten multi-index columns
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        result = result.reset_index()

        return result

    def aggregate_by_hour_and_dow(
        self,
        ts: pd.DataFrame,
        agg_funcs: List[str] = ["mean", "count"],
    ) -> pd.DataFrame:
        """
        Aggregate by both hour and day of week.

        Args:
            ts: Time-series DataFrame
            agg_funcs: Aggregation functions to apply

        Returns:
            DataFrame aggregated by (hour, day_of_week)
        """
        if ts.empty or "timestamp" not in ts.columns:
            return pd.DataFrame()

        # Extract temporal features
        df = ts.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

        # Aggregate numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in ["hour", "day_of_week"]
        ]

        if not numeric_cols:
            return pd.DataFrame()

        # Group and aggregate
        result = df.groupby(["hour", "day_of_week"])[numeric_cols].agg(agg_funcs)

        # Flatten multi-index columns
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        result = result.reset_index()

        return result