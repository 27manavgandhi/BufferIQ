"""Voting ensemble implementation."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.base import BaseEnsemble

logger = get_logger(__name__)


class VotingEnsemble(BaseEnsemble):
    """
    Voting ensemble for regression.

    Combines predictions from multiple models using weighted averaging.
    Supports soft voting (weighted average of predictions).

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from xgboost import XGBRegressor
        >>>
        >>> models = [
        ...     RandomForestRegressor(n_estimators=100),
        ...     XGBRegressor(n_estimators=100)
        ... ]
        >>>
        >>> ensemble = VotingEnsemble(models, weights=[0.6, 0.4])
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseEstimator],
        weights: Optional[np.ndarray] = None,
        voting: str = "soft",
    ) -> None:
        """
        Initialize voting ensemble.

        Args:
            base_models: List of fitted base models
            weights: Optional weights for each model (must sum to 1.0)
                    If None, uniform weights are used
            voting: Voting method ('soft' for weighted average)

        Raises:
            ValueError: If weights are invalid or models list is empty

        Example:
            >>> ensemble = VotingEnsemble(
            ...     base_models=[model1, model2, model3],
            ...     weights=np.array([0.5, 0.3, 0.2])
            ... )
        """
        super().__init__()

        if not base_models:
            raise ValueError("base_models cannot be empty")

        self.base_models = base_models
        self.voting = voting

        # Set weights
        if weights is None:
            self.weights = np.ones(len(base_models)) / len(base_models)
        else:
            self.weights = np.array(weights)

        # Validate weights
        self._validate_weights()

        logger.info(
            f"VotingEnsemble initialized with {len(base_models)} models, "
            f"weights={self.weights}"
        )

    def _validate_weights(self) -> None:
        """
        Validate ensemble weights.

        Raises:
            ValueError: If weights are invalid
        """
        if len(self.weights) != len(self.base_models):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of models ({len(self.base_models)})"
            )

        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(self.weights):.4f}")

        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """
        Fit voting ensemble.

        Note: Base models are assumed to be already fitted.
        This is a no-op that marks the ensemble as fitted.

        Args:
            X: Training features (not used, for API compatibility)
            y: Training targets (not used, for API compatibility)

        Returns:
            self: Fitted ensemble
        """
        self.validate_inputs(X, y)
        self._is_fitted = True
        logger.info("VotingEnsemble marked as fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted voting.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Weighted average predictions, shape (n_samples,)

        Raises:
            ValueError: If ensemble is not fitted

        Example:
            >>> predictions = ensemble.predict(X_test)
            >>> print(predictions.shape)
            (100,)
        """
        self.check_is_fitted()
        self.validate_inputs(X)

        # Get predictions from all base models
        predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Weighted average
        weighted_pred = np.average(predictions, axis=1, weights=self.weights)

        logger.debug(f"Generated predictions for {len(X)} samples")

        return weighted_pred

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VotingEnsemble(n_models={len(self.base_models)}, "
            f"voting={self.voting}, weights={self.weights})"
        )
