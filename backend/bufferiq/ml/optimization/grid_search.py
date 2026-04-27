"""Grid search hyperparameter optimization."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.base import BaseOptimizer

logger = get_logger(__name__)


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid search hyperparameter optimization.

    Performs exhaustive search over all combinations in a parameter grid
    using cross-validation to find the best hyperparameters.
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: dict[str, list[Any]],
        cv: int = 5,
        scoring: str = "r2",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> None:
        """
        Initialize grid search optimizer.

        Args:
            model: Scikit-learn compatible model to optimize
            param_grid: Dictionary mapping parameter names to lists of values
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Verbosity level

        Raises:
            ValueError: If param_grid is empty

        Example:
            >>> param_grid = {
            ...     'learning_rate': [0.01, 0.1, 0.2],
            ...     'max_depth': [3, 5, 7]
            ... }
            >>> optimizer = GridSearchOptimizer(model, param_grid)
            >>> results = optimizer.search(X_train, y_train)
        """
        super().__init__(model, cv, scoring, n_jobs, random_state, verbose)

        if not param_grid:
            raise ValueError("Parameter grid cannot be empty")

        self.param_grid = param_grid

        # Calculate total combinations
        self.total_combinations = int(np.prod([len(v) for v in param_grid.values()]))

        logger.info(
            f"Grid search initialized with {len(param_grid)} parameters, "
            f"{self.total_combinations} total combinations"
        )

    def search(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.

        Exhaustively searches all combinations in the parameter grid using
        cross-validation to find the best hyperparameters.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            Dictionary containing:
                - best_params (Dict): Best hyperparameters found
                - best_score (float): Best cross-validation score
                - cv_results (Dict): Detailed results for all trials
                - total_trials (int): Number of combinations tested
                - random_state (int): Random seed used

        Raises:
            ValueError: If X and y have incompatible shapes

        Example:
            >>> optimizer = GridSearchOptimizer(model, param_grid)
            >>> results = optimizer.search(X_train, y_train)
            >>> print(f"Best R²: {results['best_score']:.4f}")
            Best R²: 0.7612
        """
        # Validate inputs
        self.validate_inputs(X, y)

        logger.info(f"Starting grid search with {self.total_combinations} combinations")

        try:
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
            )

            # Fit grid search
            grid_search.fit(X, y)

            # Extract results
            self._best_params = grid_search.best_params_
            self._best_score = grid_search.best_score_
            self._search_results = {
                "best_params": self._best_params,
                "best_score": self._best_score,
                "cv_results": grid_search.cv_results_,
                "total_trials": len(grid_search.cv_results_["params"]),
                "random_state": self.random_state,
            }

            logger.info(f"Grid search complete. Best score: {self._best_score:.4f}")
            logger.info(f"Best params: {self._best_params}")

            return self._search_results

        except Exception as e:
            logger.error(f"Grid search failed: {e}", exc_info=True)
            raise
