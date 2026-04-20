"""Feature scaling utilities."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

ScalerMethod = Literal["standard", "minmax", "robust"]


class FeatureScaler:
    """Scale features for ML models."""

    def __init__(self, method: ScalerMethod = "standard") -> None:
        """
        Initialize feature scaler.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')

        Example:
            >>> scaler = FeatureScaler(method="standard")
            >>> scaler.fit(X_train, feature_cols)
            >>> X_scaled = scaler.transform(X_test, feature_cols)
        """
        self.method = method
        self._scaler: StandardScaler | MinMaxScaler | RobustScaler

        if method == "standard":
            self._scaler = StandardScaler()
        elif method == "minmax":
            self._scaler = MinMaxScaler()
        elif method == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(
                f"Invalid scaling method: {method}. "
                f"Choose from: 'standard', 'minmax', 'robust'"
            )

        self._feature_columns: list[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "FeatureScaler":
        """
        Fit scaler on training data.

        Args:
            df: Training DataFrame
            feature_columns: List of feature columns to scale

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If feature columns not in DataFrame
        """
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        self._feature_columns = feature_columns
        self._scaler.fit(df[feature_columns])
        self._is_fitted = True

        logger.info(f"Fitted {self.method} scaler on {len(feature_columns)} features")

        return self

    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """
        Transform features using fitted scaler.

        Args:
            df: DataFrame to transform
            feature_columns: List of feature columns to scale

        Returns:
            DataFrame with scaled features

        Raises:
            ValueError: If scaler not fitted or columns mismatch
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")

        if feature_columns != self._feature_columns:
            raise ValueError(
                f"Feature columns mismatch. Expected: {self._feature_columns}, "
                f"got: {feature_columns}"
            )

        result = df.copy()
        result[feature_columns] = self._scaler.transform(df[feature_columns])

        logger.info(f"Transformed {len(feature_columns)} features")

        return result

    def fit_transform(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform
            feature_columns: List of feature columns to scale

        Returns:
            DataFrame with scaled features
        """
        self.fit(df, feature_columns)
        return self.transform(df, feature_columns)

    def inverse_transform(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """
        Reverse scaling transformation.

        Args:
            df: DataFrame with scaled features
            feature_columns: List of feature columns to inverse scale

        Returns:
            DataFrame with original scale features

        Raises:
            ValueError: If scaler not fitted
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        result = df.copy()
        result[feature_columns] = self._scaler.inverse_transform(df[feature_columns])

        logger.info(f"Inverse transformed {len(feature_columns)} features")

        return result

    def save(self, path: str) -> None:
        """
        Save fitted scaler to disk.

        Args:
            path: File path to save scaler (e.g., 'scaler.joblib')

        Raises:
            ValueError: If scaler not fitted
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted scaler")

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save scaler and metadata
        joblib.dump(
            {
                "scaler": self._scaler,
                "method": self.method,
                "feature_columns": self._feature_columns,
            },
            path,
        )

        logger.info(f"Saved scaler to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """
        Load fitted scaler from disk.

        Args:
            path: File path to load scaler from

        Returns:
            Loaded FeatureScaler instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        data = joblib.load(path)

        scaler = cls(method=data["method"])
        scaler._scaler = data["scaler"]
        scaler._feature_columns = data["feature_columns"]
        scaler._is_fitted = True

        logger.info(f"Loaded scaler from {path}")

        return scaler
