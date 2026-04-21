"""Random search hyperparameter optimization."""

from typing import Any, Dict, Union

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.base import BaseOptimizer

logger = get_logger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random search hyperparameter optimization.
    
    Samples random combinations from parameter distributions using
    cross-validation to find good hyperparameters efficiently.
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        scoring: str = "r2",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> None:
        """
        Initialize random search optimizer.
        
        Args:
            model: Scikit-learn compatible model to optimize
            param_distributions: Dictionary mapping parameter names to distributions
            n_iter: Number of random parameter combinations to try
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Verbosity level
        
        Raises:
            ValueError: If param_distributions is empty or n_iter < 1
        
        Example:
            >>> from scipy.stats import loguniform, randint
            >>> param_distributions = {
            ...     'learning_rate': loguniform(0.01, 0.3),
            ...     'max_depth': randint(3, 10)
            ... }
            >>> optimizer = RandomSearchOptimizer(
            ...     model, param_distributions, n_iter=50
            ... )
            >>> results = optimizer.search(X_train, y_train)
        """
        super().__init__(model, cv, scoring, n_jobs, random_state, verbose)
        
        if not param_distributions:
            raise ValueError("Parameter distributions cannot be empty")
        
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        
        logger.info(
            f"Random search initialized with {len(param_distributions)} "
            f"parameters, {n_iter} iterations"
        )

    def search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Samples random parameter combinations from distributions and evaluates
        them using cross-validation.
        
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
            >>> optimizer = RandomSearchOptimizer(model, distributions, n_iter=50)
            >>> results = optimizer.search(X_train, y_train)
            >>> print(f"Best R²: {results['best_score']:.4f}")
            Best R²: 0.7534
        """
        # Validate inputs
        self.validate_inputs(X, y)
        
        logger.info(f"Starting random search with {self.n_iter} iterations")
        
        try:
            # Create RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True,
            )
            
            # Fit random search
            random_search.fit(X, y)
            
            # Extract results
            self._best_params = random_search.best_params_
            self._best_score = random_search.best_score_
            self._search_results = {
                "best_params": self._best_params,
                "best_score": self._best_score,
                "cv_results": random_search.cv_results_,
                "total_trials": len(random_search.cv_results_["params"]),
                "random_state": self.random_state,
            }
            
            logger.info(
                f"Random search complete. Best score: {self._best_score:.4f}"
            )
            logger.info(f"Best params: {self._best_params}")
            
            return self._search_results
            
        except Exception as e:
            logger.error(f"Random search failed: {e}", exc_info=True)
            raise