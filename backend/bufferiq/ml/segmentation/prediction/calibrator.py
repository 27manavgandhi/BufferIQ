"""Model calibration for accurate confidence intervals."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ModelCalibrator:
    """Calibrate predictions for accurate confidence intervals."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize model calibrator."""
        self.config = config or {}
        self.calibrator: Any = None
        self.is_fitted = False

    def fit(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> None:
        """
        Fit calibration model.

        Args:
            predicted: Predicted values
            actual: Actual values
        """
        self.calibrator = IsotonicRegression(bounds=(0, 1), out_of_bounds="clip")
        self.calibrator.fit(predicted, actual)
        self.is_fitted = True

    def calibrate(self, predicted: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions.

        Args:
            predicted: Predicted values

        Returns:
            Calibrated predictions

        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.calibrator.predict(predicted)

    def get_confidence_interval(
        self, calibrated_prediction: float, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Get confidence interval for a prediction.

        Args:
            calibrated_prediction: Calibrated prediction
            confidence_level: Confidence level (default 0.95)

        Returns:
            (lower_bound, upper_bound)
        """
        # Calculate margin of error
        z_score = self._get_z_score(confidence_level)
        margin = z_score * 0.1  # Assume 10% standard error

        lower = max(calibrated_prediction - margin, 0.0)
        upper = min(calibrated_prediction + margin, 1.0)

        return (lower, upper)

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return z_scores.get(confidence_level, 1.96)