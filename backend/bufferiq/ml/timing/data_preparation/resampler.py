"""Resample and fill missing timestamps in time-series."""

import pandas as pd
from typing import Literal


class TimeSeriesResampler:
    """Fill missing timestamps in time-series data."""

    def __init__(self) -> None:
        """Initialize TimeSeriesResampler."""
        pass

    def resample_hourly(
        self,
        ts: pd.DataFrame,
        method: Literal["linear", "forward", "backward", "zero"] = "linear",
    ) -> pd.DataFrame:
        """
        Resample to hourly frequency and fill missing values.

        Args:
            ts: Time-series DataFrame with 'timestamp' column
            method: Interpolation method ('linear', 'forward', 'backward', 'zero')

        Returns:
            Resampled DataFrame with hourly frequency

        Example:
            >>> resampler = TimeSeriesResampler()
            >>> ts_filled = resampler.resample_hourly(ts, method='linear')
        """
        if ts.empty or "timestamp" not in ts.columns:
            return ts

        # Set timestamp as index
        df = ts.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Resample to hourly
        df_resampled = df.resample("H").mean()

        # Fill missing values based on method
        df_filled = self._fill_missing(df_resampled, method)

        # Reset index
        df_filled = df_filled.reset_index()

        return df_filled

    def fill_missing(
        self,
        ts: pd.DataFrame,
        method: Literal["forward", "backward", "linear", "zero"] = "forward",
    ) -> pd.DataFrame:
        """
        Fill missing values in time-series.

        Args:
            ts: Time-series DataFrame
            method: Fill method

        Returns:
            DataFrame with missing values filled
        """
        if ts.empty:
            return ts

        df = ts.copy()

        # Identify numeric columns (exclude timestamp)
        numeric_cols = df.select_dtypes(include=["number"]).columns

        # Fill based on method
        if method == "forward":
            df[numeric_cols] = df[numeric_cols].fillna(method="ffill")
        elif method == "backward":
            df[numeric_cols] = df[numeric_cols].fillna(method="bfill")
        elif method == "linear":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif method == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0.0)
        else:
            raise ValueError(f"Unknown fill method: {method}")

        # Fill any remaining NaNs with 0
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        return df

    def _fill_missing(
        self,
        df: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        """Internal method to fill missing values."""
        if method == "linear":
            return df.interpolate(method="linear")
        elif method == "forward":
            return df.fillna(method="ffill")
        elif method == "backward":
            return df.fillna(method="bfill")
        elif method == "zero":
            return df.fillna(0.0)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")