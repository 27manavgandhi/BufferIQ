"""Forecast audience activity using Prophet."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ProphetForecaster:
    """Forecast audience activity using Facebook Prophet."""

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        interval_width: float = 0.80,
    ) -> None:
        """
        Initialize ProphetForecaster.

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Strength of seasonality
            daily_seasonality: Include daily seasonality
            weekly_seasonality: Include weekly seasonality
            yearly_seasonality: Include yearly seasonality
            interval_width: Confidence interval width (0.80 = 80%)
        """
        if Prophet is None:
            raise ImportError(
                "Prophet is required for forecasting. Install with: pip install prophet"
            )

        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.interval_width = interval_width

        self.model: Optional[Prophet] = None
        self.is_fitted = False

    def fit(self, ts: pd.DataFrame) -> None:
        """
        Fit Prophet model on time-series.

        Args:
            ts: Time-series with 'timestamp' and 'engagement_score' columns

        Example:
            >>> forecaster = ProphetForecaster()
            >>> forecaster.fit(ts)
            >>> assert forecaster.is_fitted
        """
        if ts.empty:
            raise ValueError("Cannot fit on empty time-series")

        if "timestamp" not in ts.columns or "engagement_score" not in ts.columns:
            raise ValueError("Time-series must have 'timestamp' and 'engagement_score'")

        # Prepare data for Prophet (requires 'ds' and 'y')
        df_prophet = pd.DataFrame(
            {
                "ds": pd.to_datetime(ts["timestamp"]),
                "y": ts["engagement_score"],
            }
        )

        # Remove any NaNs
        df_prophet = df_prophet.dropna()

        if len(df_prophet) < 2:
            raise ValueError("Need at least 2 valid data points to fit Prophet")

        # Create and fit model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width,
        )

        # Suppress Prophet's verbose output
        import logging

        logging.getLogger("prophet").setLevel(logging.ERROR)

        self.model.fit(df_prophet)
        self.is_fitted = True

    def predict(self, periods: int = 168) -> pd.DataFrame:
        """
        Predict future engagement.

        Args:
            periods: Number of hours to forecast (default: 168 = 7 days)

        Returns:
            DataFrame with ds, yhat, yhat_lower, yhat_upper

        Example:
            >>> forecast = forecaster.predict(periods=24)
            >>> assert len(forecast) == 24
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq="H")

        # Predict
        forecast = self.model.predict(future)

        # Return only future predictions
        forecast = forecast.tail(periods)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def predict_at_times(self, timestamps: List[datetime]) -> np.ndarray:
        """
        Predict engagement at specific timestamps.

        Args:
            timestamps: List of datetime objects

        Returns:
            Array of predicted engagement scores

        Example:
            >>> from datetime import datetime, timedelta
            >>> times = [datetime.now() + timedelta(hours=i) for i in range(24)]
            >>> predictions = forecaster.predict_at_times(times)
            >>> assert len(predictions) == 24
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create dataframe with timestamps
        df = pd.DataFrame({"ds": pd.to_datetime(timestamps)})

        # Predict
        forecast = self.model.predict(df)

        return forecast["yhat"].values

    def get_components(self) -> pd.DataFrame:
        """
        Get trend and seasonality components.

        Returns:
            DataFrame with trend, weekly, daily components
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get in-sample predictions
        future = self.model.make_future_dataframe(periods=0, freq="H")
        forecast = self.model.predict(future)

        # Select component columns
        components = ["ds", "trend"]
        if self.weekly_seasonality and "weekly" in forecast.columns:
            components.append("weekly")
        if self.daily_seasonality and "daily" in forecast.columns:
            components.append("daily")

        return forecast[components]