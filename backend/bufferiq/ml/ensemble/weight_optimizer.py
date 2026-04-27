"""Optimize ensemble weights."""

from typing import Any, Literal

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

OptimizationMethod = Literal["uniform", "performance", "optuna", "grid"]


class WeightOptimizer:
    """
    Optimize ensemble weights using various methods.

    Supports:
    - uniform: Equal weights (1/n)
    - performance: Weights proportional to R² scores
    - optuna: Optimized weights using Optuna
    - grid: Grid search over weight space

    Example:
        >>> optimizer = WeightOptimizer(
        ...     base_models=[model1, model2, model3],
        ...     method='optuna'
        ... )
        >>> optimal_weights = optimizer.optimize(X_train, y_train)
        >>> print(f"Optimal weights: {optimal_weights}")
    """

    def __init__(
        self,
        base_models: list[BaseEstimator],
        method: OptimizationMethod = "optuna",
        cv: int = 5,
        scoring: str = "r2",
        n_trials: int = 100,
        random_state: int = 42,
    ) -> None:
        """
        Initialize weight optimizer.

        Args:
            base_models: List of fitted base models
            method: Optimization method
            cv: Cross-validation folds
            scoring: Scoring metric
            n_trials: Number of Optuna trials (for optuna method)
            random_state: Random seed

        Raises:
            ValueError: If base_models is empty

        Example:
            >>> optimizer = WeightOptimizer(
            ...     base_models=[model1, model2],
            ...     method='optuna',
            ...     n_trials=50
            ... )
        """
        if not base_models:
            raise ValueError("base_models cannot be empty")

        self.base_models = base_models
        self.method = method
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.random_state = random_state

        logger.info(
            f"WeightOptimizer initialized: method={method}, "
            f"n_models={len(base_models)}"
        )

    def _uniform_weights(self) -> np.ndarray:
        """
        Compute uniform weights.

        Returns:
            Uniform weights array
        """
        n_models = len(self.base_models)
        weights = np.ones(n_models) / n_models
        logger.info(f"Uniform weights: {weights}")
        return weights

    def _performance_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute performance-based weights.

        Args:
            X: Features
            y: Targets

        Returns:
            Performance-based weights
        """
        logger.info("Computing performance-based weights")

        performances = []
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            r2 = r2_score(y, pred)
            performances.append(max(r2, 0.0))
            logger.debug(f"Model {i+1} R²: {r2:.4f}")

        performances_array = np.array(performances)

        total = np.sum(performances_array)
        if total > 0:
            weights = performances_array / total
        else:
            logger.warning("All models have R² <= 0, using uniform weights")
            weights = self._uniform_weights()

        logger.info(f"Performance weights: {weights}")
        return weights

    def _optuna_weights(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Optimize weights using Optuna.

        Args:
            X: Features
            y: Targets

        Returns:
            Dictionary with optimal weights and best score
        """
        logger.info(f"Optimizing weights using Optuna ({self.n_trials} trials)")

        self.X = X
        self.y = y

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        study.optimize(
            self._optuna_objective, n_trials=self.n_trials, show_progress_bar=False
        )

        # Extract optimal weights
        optimal_weights = []
        for i in range(len(self.base_models) - 1):
            optimal_weights.append(study.best_params[f"weight_{i}"])
        optimal_weights.append(1.0 - sum(optimal_weights))

        optimal_weights_array = np.array(optimal_weights)

        logger.info(
            f"Optuna optimization complete: "
            f"best_score={study.best_value:.4f}, "
            f"optimal_weights={optimal_weights_array}"
        )

        return {
            "weights": optimal_weights_array,
            "best_score": study.best_value,
            "study": study,
        }

    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Cross-validation score
        """
        # Suggest weights (must sum to 1.0)
        n_models = len(self.base_models)
        weights = []

        for i in range(n_models - 1):
            w = trial.suggest_float(f"weight_{i}", 0.0, 1.0)
            weights.append(w)

        # Last weight is constrained
        last_weight = 1.0 - sum(weights)

        if last_weight < 0:
            return -999.0

        weights.append(last_weight)
        weights_array = np.array(weights)

        # Get predictions from all models
        predictions = np.column_stack(
            [model.predict(self.X) for model in self.base_models]
        )

        # Weighted average
        ensemble_pred = np.average(predictions, axis=1, weights=weights_array)

        # Calculate R²
        score = r2_score(self.y, ensemble_pred)

        return score

    def _grid_weights(
        self, X: np.ndarray, y: np.ndarray, grid_size: int = 10
    ) -> dict[str, Any]:
        """
        Optimize weights using grid search.

        Args:
            X: Features
            y: Targets
            grid_size: Number of points per dimension

        Returns:
            Dictionary with optimal weights and best score
        """
        logger.info(f"Grid search over weights (grid_size={grid_size})")

        n_models = len(self.base_models)

        # For simplicity, only support 2-3 models with grid search
        if n_models > 3:
            logger.warning(
                f"Grid search with {n_models} models is expensive, "
                f"falling back to Optuna"
            )
            return self._optuna_weights(X, y)

        # Get predictions from all models
        predictions = np.column_stack([model.predict(X) for model in self.base_models])

        best_score = -np.inf
        best_weights = None

        # Generate weight grid
        weight_points = np.linspace(0, 1, grid_size)

        if n_models == 2:
            for w1 in weight_points:
                w2 = 1.0 - w1
                weights = np.array([w1, w2])

                ensemble_pred = np.average(predictions, axis=1, weights=weights)
                score = r2_score(y, ensemble_pred)

                if score > best_score:
                    best_score = score
                    best_weights = weights

        elif n_models == 3:
            for w1 in weight_points:
                for w2 in weight_points:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0:
                        continue

                    weights = np.array([w1, w2, w3])

                    ensemble_pred = np.average(predictions, axis=1, weights=weights)
                    score = r2_score(y, ensemble_pred)

                    if score > best_score:
                        best_score = score
                        best_weights = weights

        logger.info(
            f"Grid search complete: best_score={best_score:.4f}, "
            f"optimal_weights={best_weights}"
        )

        return {
            "weights": best_weights,
            "best_score": best_score,
        }

    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Optimize weights using specified method.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Optimal weights array

        Example:
            >>> weights = optimizer.optimize(X_train, y_train)
            >>> print(f"Optimal weights: {weights}")
        """
        if self.method == "uniform":
            return self._uniform_weights()

        elif self.method == "performance":
            return self._performance_weights(X, y)

        elif self.method == "optuna":
            result = self._optuna_weights(X, y)
            return result["weights"]

        elif self.method == "grid":
            result = self._grid_weights(X, y)
            return result["weights"]

        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def optimize_with_details(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Optimize weights and return detailed results.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Dictionary with weights, score, and method-specific details

        Example:
            >>> results = optimizer.optimize_with_details(X_train, y_train)
            >>> print(f"Best score: {results['best_score']:.4f}")
        """
        if self.method == "uniform":
            weights = self._uniform_weights()
            return {"weights": weights, "method": "uniform"}

        elif self.method == "performance":
            weights = self._performance_weights(X, y)
            return {"weights": weights, "method": "performance"}

        elif self.method == "optuna":
            return self._optuna_weights(X, y)

        elif self.method == "grid":
            return self._grid_weights(X, y)

        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
