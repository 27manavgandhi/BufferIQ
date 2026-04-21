"""Base optimizer abstract class for hyperparameter optimization."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    
    Provides common interface and validation logic for Grid Search,
    Random Search, and Bayesian optimization.
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv: int = 5,
        scoring: str = "r2",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> None:
        """
        Initialize base optimizer.
        
        Args:
            model: Scikit-learn compatible model to optimize
            cv: Number of cross-validation folds
            scoring: Scoring metric ('r2', 'neg_mean_absolute_error', etc.)
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Raises:
            ValueError: If cv < 2
        """
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")
        
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        logger.info(
            f"Initialized {self.__class__.__name__} with cv={cv}, "
            f"scoring={scoring}, random_state={random_state}"
        )

    def validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate training data inputs.
        
        Args:
            X: Training features
            y: Training targets
        
        Raises:
            ValueError: If inputs are invalid
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty input arrays provided")
        
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length: {len(X)} != {len(y)}"
            )
        
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")
        
        if np.any(np.isinf(y)):
            raise ValueError("y contains infinite values")
        
        logger.debug(f"Input validation passed: X={X.shape}, y={y.shape}")

    @abstractmethod
    def search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
        
        Returns:
            Dictionary containing:
                - best_params (Dict): Best hyperparameters found
                - best_score (float): Best cross-validation score
                - cv_results (Dict): Detailed results for all trials
                - total_trials (int): Number of trials performed
        
        Raises:
            ValueError: If inputs are invalid
        """
        pass

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get best hyperparameters from last search.
        
        Returns:
            Best parameters found, or None if search not run yet
        """
        if not hasattr(self, "_best_params"):
            logger.warning("No search has been performed yet")
            return None
        return self._best_params

    def get_best_score(self) -> Optional[float]:
        """
        Get best score from last search.
        
        Returns:
            Best cross-validation score, or None if search not run yet
        """
        if not hasattr(self, "_best_score"):
            logger.warning("No search has been performed yet")
            return None
        return self._best_score

    def get_search_results(self) -> Optional[Dict[str, Any]]:
        """
        Get complete search results from last search.
        
        Returns:
            Full search results dictionary, or None if search not run yet
        """
        if not hasattr(self, "_search_results"):
            logger.warning("No search has been performed yet")
            return None
        return self._search_results