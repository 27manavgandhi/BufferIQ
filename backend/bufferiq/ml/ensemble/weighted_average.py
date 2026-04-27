"""Weighted average ensemble implementation."""

from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.base import BaseEnsemble

logger = get_logger(__name__)

WeightMethod = Literal["uniform", "performance", "optimized"]


class WeightedAverageEnsemble(BaseEnsemble):
    """
    Weighted average ensemble.

    Simple weighted average of predictions with various weighting strategies:
    - uniform: Equal weights (1/n)
    - performance: Weights proportional to R² scores
    - optimized: Weights optimized using provided optimizer

    Example:
        >>> ensemble = WeightedAverageEnsemble(
        ...     base_models=[model1, model2, model3],
        ...     weight_method='performance'
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseEstimator],
        weight_method: WeightMethod = "uniform",
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize weighted average ensemble.

        Args:
            base_models: List of fitted base models
            weight_method: Method for computing weights
            weights: Optional pre-computed weights (overrides weight_method)

        Raises:
            ValueError: If base_models is empty or weights invalid

        Example:
            >>> ensemble = WeightedAverageEnsemble(
            ...     base_models=[model1, model2],
            ...     weight_method='performance'
            ... )
        """
        super().__init__()

        if not base_models:
            raise ValueError("base_models cannot be empty")

        self.base_models = base_models
        self.weight_method = weight_method
        self._user_weights = weights
        self.weights: Optional[np.ndarray] = None

        logger.info(
            f"WeightedAverageEnsemble initialized with {len(base_models)} models, "
            f"weight_method={weight_method}"
        )

    def _compute_uniform_weights(self) -> np.ndarray:
        """
        Compute uniform weights (equal for all models).

        Returns:
            Uniform weights array
        """
        n_models = len(self.base_models)
        weights = np.ones(n_models) / n_models
        logger.info(f"Computed uniform weights: {weights}")
        return weights

    def _compute_performance_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute performance-based weights (proportional to R²).

        Args:
            X: Validation features
            y: Validation targets

        Returns:
            Performance-based weights
        """
        logger.info("Computing performance-based weights")

        performances = []
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            r2 = r2_score(y, pred)
            performances.append(max(r2, 0.0))  # Ensure non-negative
            logger.debug(f"Model {i+1} R²: {r2:.4f}")

        performances_array = np.array(performances)

        # Normalize to sum to 1.0
        total = np.sum(performances_array)
        if total > 0:
            weights = performances_array / total
        else:
            logger.warning("All models have R² <= 0, using uniform weights")
            weights = self._compute_uniform_weights()

        logger.info(f"Computed performance weights: {weights}")
        return weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedAverageEnsemble":
        """
        Fit weighted average ensemble.

        Computes weights based on specified method.

        Args:
            X: Training features (used for performance weighting)
            y: Training targets (used for performance weighting)

        Returns:
            self: Fitted ensemble

        Example:
            >>> ensemble.fit(X_train, y_train)
        """
        self.validate_inputs(X, y)

        # Use provided weights if available
        if self._user_weights is not None:
            self.weights = np.array(self._user_weights)
            logger.info(f"Using provided weights: {self.weights}")
        elif self.weight_method == "uniform":
            self.weights = self._compute_uniform_weights()
        elif self.weight_method == "performance":
            self.weights = self._compute_performance_weights(X, y)
        elif self.weight_method == "optimized":
            # Weights should be set externally by optimizer
            if self.weights is None:
                logger.warning(
                    "weight_method='optimized' but no weights set, "
                    "using uniform weights"
                )
                self.weights = self._compute_uniform_weights()
        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")

        # Validate weights
        self._validate_weights()

        self._is_fitted = True
        logger.info("WeightedAverageEnsemble training complete")

        return self

    def _validate_weights(self) -> None:
        """
        Validate ensemble weights.

        Raises:
            ValueError: If weights are invalid
        """
        if self.weights is None:
            raise ValueError("Weights not set")

        if len(self.weights) != len(self.base_models):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of models ({len(self.base_models)})"
            )

        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(self.weights):.4f}")

        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative")

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set ensemble weights manually.

        Args:
            weights: Weight array (must sum to 1.0)

        Example:
            >>> ensemble.set_weights(np.array([0.5, 0.3, 0.2]))
        """
        self.weights = np.array(weights)
        self._validate_weights()
        logger.info(f"Weights set to: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted average.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Weighted average predictions, shape (n_samples,)

        Raises:
            ValueError: If ensemble is not fitted

        Example:
            >>> predictions = ensemble.predict(X_test)
        """
        self.check_is_fitted()
        self.validate_inputs(X)

        if self.weights is None:
            raise ValueError("Weights not computed. Call fit() first.")

        # Get predictions from all base models
        predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Weighted average
        weighted_pred = np.average(predictions, axis=1, weights=self.weights)

        logger.debug(f"Generated predictions for {len(X)} samples")

        return weighted_pred

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WeightedAverageEnsemble(n_models={len(self.base_models)}, "
            f"weight_method={self.weight_method}, weights={self.weights})"
        )
