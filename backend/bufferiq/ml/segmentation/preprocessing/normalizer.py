"""Data normalization utilities."""

from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataNormalizer:
    """Normalize and scale feature data."""

    def __init__(self, method: str = "standard") -> None:
        """
        Initialize normalizer.

        Args:
            method: "standard" or "minmax"

        Raises:
            ValueError: If invalid method
        """
        if method not in ["standard", "minmax"]:
            raise ValueError(f"Invalid method: {method}. Must be 'standard' or 'minmax'")

        self.method = method
        self.scaler: Any = StandardScaler() if method == "standard" else MinMaxScaler()
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        """
        Fit scaler to data.

        Args:
            data: Feature matrix (n_samples, n_features)
        """
        if data.shape[0] == 0:
            raise ValueError("Cannot fit scaler to empty data")

        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data.

        Args:
            data: Feature matrix

        Returns:
            Normalized data

        Raises:
            ValueError: If scaler not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return self.scaler.inverse_transform(data)