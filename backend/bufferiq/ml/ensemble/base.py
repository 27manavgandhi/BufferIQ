"""Base class for all ensemble models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class BaseEnsemble(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract base class for all ensemble models.

    Provides common interface and utilities for ensemble implementations.
    All ensemble models must implement fit() and predict() methods.
    """

    def __init__(self) -> None:
        """Initialize base ensemble."""
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEnsemble":
        """
        Fit the ensemble model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            self: Fitted ensemble

        Raises:
            ValueError: If X and y have incompatible shapes
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)

        Raises:
            ValueError: If ensemble is not fitted
        """
        pass

    def validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input arrays.

        Args:
            X: Feature matrix
            y: Optional target vector

        Raises:
            ValueError: If inputs are invalid
        """
        if X.size == 0:
            raise ValueError("Empty input X")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y is not None:
            if y.size == 0:
                raise ValueError("Empty input y")

            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

    def check_is_fitted(self) -> None:
        """
        Check if ensemble is fitted.

        Raises:
            ValueError: If ensemble is not fitted
        """
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before making predictions"
            )

    def save(self, path: Path) -> None:
        """
        Save ensemble to disk.

        Args:
            path: Path to save ensemble

        Example:
            >>> ensemble.save(Path('outputs/models/ensembles/stacking_v1.0.0.joblib'))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseEnsemble":
        """
        Load ensemble from disk.

        Args:
            path: Path to load ensemble from

        Returns:
            Loaded ensemble

        Raises:
            FileNotFoundError: If path does not exist

        Example:
            >>> ensemble = BaseEnsemble.load(Path('outputs/models/ensembles/stacking_v1.0.0.joblib'))
        """
        if not path.exists():
            raise FileNotFoundError(f"Ensemble not found at {path}")

        ensemble = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")
        return ensemble

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: If True, return parameters for sub-estimators

        Returns:
            Parameter names mapped to their values
        """
        params = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                params[key] = getattr(self, key)
        return params

    def set_params(self, **params: Any) -> "BaseEnsemble":
        """
        Set parameters for this estimator.

        Args:
            **params: Estimator parameters

        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """String representation of ensemble."""
        return f"{self.__class__.__name__}()"
