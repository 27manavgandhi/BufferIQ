"""Intelligent model selection for ensembles."""


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.diversity_analyzer import DiversityAnalyzer

logger = get_logger(__name__)


class ModelSelector:
    """
    Select diverse, high-performing models for ensemble.

    Uses greedy selection algorithm to choose models that:
    1. Meet minimum performance threshold
    2. Maximize ensemble diversity
    3. Improve overall ensemble performance

    Example:
        >>> selector = ModelSelector(
        ...     min_performance=0.70,
        ...     min_diversity=0.10,
        ...     max_models=5
        ... )
        >>> selected_indices = selector.select(models, X_val, y_val)
        >>> selected_models = [models[i] for i in selected_indices]
    """

    def __init__(
        self,
        min_performance: float = 0.70,
        min_diversity: float = 0.10,
        max_models: int = 5,
    ) -> None:
        """
        Initialize model selector.

        Args:
            min_performance: Minimum R² to include a model
            min_diversity: Minimum correlation diversity required
            max_models: Maximum number of models in ensemble

        Raises:
            ValueError: If parameters are invalid

        Example:
            >>> selector = ModelSelector(min_performance=0.75, max_models=3)
        """
        if not 0.0 <= min_performance <= 1.0:
            raise ValueError(
                f"min_performance must be in [0, 1], got {min_performance}"
            )

        if not 0.0 <= min_diversity <= 1.0:
            raise ValueError(f"min_diversity must be in [0, 1], got {min_diversity}")

        if max_models < 1:
            raise ValueError(f"max_models must be >= 1, got {max_models}")

        self.min_performance = min_performance
        self.min_diversity = min_diversity
        self.max_models = max_models

        logger.info(
            f"ModelSelector initialized: min_performance={min_performance}, "
            f"min_diversity={min_diversity}, max_models={max_models}"
        )

    def select(
        self,
        models: list[BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[int]:
        """
        Select models using greedy algorithm.

        Algorithm:
        1. Evaluate all models on validation set
        2. Filter by minimum performance
        3. Select best model as starting point
        4. Iteratively add most diverse model that improves ensemble

        Args:
            models: List of candidate models
            X_val: Validation features
            y_val: Validation targets

        Returns:
            List of selected model indices

        Raises:
            ValueError: If no models meet criteria

        Example:
            >>> selected = selector.select(models, X_val, y_val)
            >>> print(f"Selected {len(selected)} models")
        """
        if not models:
            raise ValueError("models list cannot be empty")

        logger.info(f"Selecting from {len(models)} candidate models")

        # Evaluate all models
        performances = []
        predictions = []

        for i, model in enumerate(models):
            pred = model.predict(X_val)
            predictions.append(pred)
            r2 = r2_score(y_val, pred)
            performances.append(r2)
            logger.debug(f"Model {i}: R² = {r2:.4f}")

        predictions_array = np.column_stack(predictions)

        # Filter by minimum performance
        candidates = [
            i for i, perf in enumerate(performances) if perf >= self.min_performance
        ]

        if len(candidates) == 0:
            raise ValueError(
                f"No models meet minimum performance threshold {self.min_performance}"
            )

        logger.info(
            f"{len(candidates)} models meet performance threshold "
            f"(>= {self.min_performance})"
        )

        # Start with best model
        best_idx = candidates[np.argmax([performances[i] for i in candidates])]
        selected = [best_idx]

        logger.info(
            f"Starting with best model: index={best_idx}, "
            f"R²={performances[best_idx]:.4f}"
        )

        # Greedy selection
        while len(selected) < self.max_models and len(selected) < len(candidates):
            best_candidate = None
            best_diversity = -1.0

            for idx in candidates:
                if idx in selected:
                    continue

                # Calculate diversity with selected models
                test_selected = selected + [idx]
                test_predictions = predictions_array[:, test_selected]
                diversity = DiversityAnalyzer.correlation_diversity(test_predictions)

                if diversity > best_diversity and diversity >= self.min_diversity:
                    best_diversity = diversity
                    best_candidate = idx

            if best_candidate is None:
                logger.info(
                    f"No more models meet diversity threshold {self.min_diversity}"
                )
                break

            selected.append(best_candidate)
            logger.info(
                f"Added model {best_candidate}: R²={performances[best_candidate]:.4f}, "
                f"diversity={best_diversity:.4f}"
            )

        logger.info(f"Selected {len(selected)} models: {selected}")

        return selected

    def select_with_details(
        self,
        models: list[BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[list[int], dict]:
        """
        Select models and return detailed information.

        Args:
            models: List of candidate models
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Tuple of (selected indices, details dict)

        Example:
            >>> selected, details = selector.select_with_details(models, X_val, y_val)
            >>> print(f"Selected models: {details['selected_performances']}")
        """
        selected = self.select(models, X_val, y_val)

        # Get predictions and performances for selected models
        selected_predictions = []
        selected_performances = []

        for idx in selected:
            pred = models[idx].predict(X_val)
            selected_predictions.append(pred)
            r2 = r2_score(y_val, pred)
            selected_performances.append(r2)

        selected_predictions_array = np.column_stack(selected_predictions)

        # Calculate diversity
        diversity = DiversityAnalyzer.correlation_diversity(selected_predictions_array)

        details = {
            "selected_indices": selected,
            "selected_performances": selected_performances,
            "avg_performance": np.mean(selected_performances),
            "diversity": diversity,
        }

        return selected, details
