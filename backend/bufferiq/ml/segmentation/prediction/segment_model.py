"""Segment-specific engagement models."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class SegmentEngagementModel:
    """Train and use engagement models for specific segments."""

    def __init__(
        self, segment_id: str, config: Dict[str, Any] | None = None
    ) -> None:
        """
        Initialize segment engagement model.

        Args:
            segment_id: Segment identifier
            config: Configuration dictionary
        """
        self.segment_id = segment_id
        self.config = config or {}
        self.model: Any = None
        self.model_type = self.config.get("model_type", "linear")
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit engagement model.

        Args:
            X: Feature matrix
            y: Engagement targets
        """
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        else:
            self.model = LinearRegression()

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict engagement.

        Args:
            X: Feature matrix

        Returns:
            Predicted engagement scores

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X)
        return np.clip(predictions, 0.0, 1.0)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)

        return None

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score model performance.

        Args:
            X: Feature matrix
            y: True engagement scores

        Returns:
            R² score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return float(self.model.score(X, y))