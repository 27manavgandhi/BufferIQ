"""Bayesian hyperparameter optimization using scikit-optimize."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.base import BaseOptimizer

logger = get_logger(__name__)

# Try to import skopt, but make it optional
try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not installed, BayesianOptimizer unavailable")


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian hyperparameter optimization using Gaussian processes.

    Uses scikit-optimize's BayesSearchCV to intelligently search the
    parameter space by learning from previous trials.

    Requires: pip install scikit-optimize
    """

    def __init__(
        self,
        model: BaseEstimator,
        search_spaces: dict[str, Any],
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "r2",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> None:
        """
        Initialize Bayesian optimizer.

        Args:
            model: Scikit-learn compatible model to optimize
            search_spaces: Dictionary mapping parameter names to skopt spaces
            n_iter: Number of iterations to perform
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Verbosity level

        Raises:
            ImportError: If scikit-optimize is not installed
            ValueError: If search_spaces is empty or n_iter < 1

        Example:
            >>> from skopt.space import Real, Integer
            >>> search_spaces = {
            ...     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            ...     'max_depth': Integer(3, 10)
            ... }
            >>> optimizer = BayesianOptimizer(model, search_spaces, n_iter=50)
            >>> results = optimizer.search(X_train, y_train)
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for BayesianOptimizer. "
                "Install it with: pip install scikit-optimize"
            )

        super().__init__(model, cv, scoring, n_jobs, random_state, verbose)

        if not search_spaces:
            raise ValueError("Search spaces cannot be empty")

        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")

        self.search_spaces = search_spaces
        self.n_iter = n_iter

        logger.info(
            f"Bayesian optimization initialized with {len(search_spaces)} "
            f"parameters, {n_iter} iterations"
        )

    def search(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Perform Bayesian hyperparameter optimization.

        Uses Gaussian processes to model the objective function and
        intelligently select the next parameter combination to try.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            Dictionary containing:
                - best_params (Dict): Best hyperparameters found
                - best_score (float): Best cross-validation score
                - cv_results (Dict): Detailed results for all trials
                - total_trials (int): Number of iterations performed
                - random_state (int): Random seed used

        Raises:
            ValueError: If X and y have incompatible shapes

        Example:
            >>> optimizer = BayesianOptimizer(model, search_spaces, n_iter=50)
            >>> results = optimizer.search(X_train, y_train)
            >>> print(f"Best R²: {results['best_score']:.4f}")
            Best R²: 0.7589
        """
        # Validate inputs
        self.validate_inputs(X, y)

        logger.info(f"Starting Bayesian optimization with {self.n_iter} iterations")

        try:
            # Create BayesSearchCV
            bayes_search = BayesSearchCV(
                estimator=self.model,
                search_spaces=self.search_spaces,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True,
            )

            # Fit Bayesian search
            bayes_search.fit(X, y)

            # Extract results
            self._best_params = bayes_search.best_params_
            self._best_score = bayes_search.best_score_
            self._search_results = {
                "best_params": self._best_params,
                "best_score": self._best_score,
                "cv_results": bayes_search.cv_results_,
                "total_trials": len(bayes_search.cv_results_["params"]),
                "random_state": self.random_state,
            }

            logger.info(
                f"Bayesian optimization complete. Best score: {self._best_score:.4f}"
            )
            logger.info(f"Best params: {self._best_params}")

            return self._search_results

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}", exc_info=True)
            raise
