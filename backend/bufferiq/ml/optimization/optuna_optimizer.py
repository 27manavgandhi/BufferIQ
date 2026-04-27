"""Optuna-based hyperparameter optimization with pruning support."""

from typing import Any, Optional

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.base import BaseOptimizer

logger = get_logger(__name__)


class OptunaOptimizer(BaseOptimizer):
    """
    Optuna-based hyperparameter optimization.

    Provides advanced features like pruning, multiple sampling strategies,
    study persistence, and resumable optimization.
    """

    def __init__(
        self,
        model: BaseEstimator,
        search_space: dict[str, dict[str, Any]],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        sampler: Optional[BaseSampler] = None,
        pruner: Optional[BasePruner] = None,
        direction: str = "maximize",
        study_name: str = "optuna_study",
        storage: Optional[str] = None,
        cv: int = 5,
        scoring: str = "r2",
        random_state: int = 42,
        verbose: int = 1,
    ) -> None:
        """
        Initialize Optuna optimizer.

        Args:
            model: Scikit-learn compatible model to optimize
            search_space: Parameter search space in Optuna format
            n_trials: Maximum number of trials to run
            timeout: Maximum optimization time in seconds
            sampler: Optuna sampler (TPESampler, RandomSampler, etc.)
            pruner: Optuna pruner (MedianPruner, HyperbandPruner, etc.)
            direction: Optimization direction ('maximize' or 'minimize')
            study_name: Name for the study
            storage: Storage URL (e.g., 'sqlite:///study.db')
            cv: Number of cross-validation folds
            scoring: Scoring metric
            random_state: Random seed for reproducibility
            verbose: Verbosity level

        Raises:
            ValueError: If direction is invalid

        Example:
            >>> from optuna.samplers import TPESampler
            >>> from optuna.pruners import MedianPruner
            >>> search_space = {
            ...     'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            ...     'max_depth': {'type': 'int', 'low': 3, 'high': 10}
            ... }
            >>> optimizer = OptunaOptimizer(
            ...     model=xgboost_model,
            ...     search_space=search_space,
            ...     n_trials=100,
            ...     sampler=TPESampler(seed=42),
            ...     pruner=MedianPruner()
            ... )
            >>> results = optimizer.search(X_train, y_train)
        """
        super().__init__(model, cv, scoring, -1, random_state, verbose)

        if direction not in ["maximize", "minimize"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'maximize' or 'minimize'"
            )

        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.study: Optional[optuna.Study] = None

        logger.info(
            f"Optuna optimizer initialized: {n_trials} trials, "
            f"direction={direction}, study_name={study_name}"
        )

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type")

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Cross-validation score (higher is better for maximize)

        Raises:
            optuna.TrialPruned: If trial should be pruned
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial)

        # Create model with suggested parameters
        model = clone(self.model).set_params(**params)

        # Perform cross-validation
        cv_scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        # Report intermediate values for pruning
        for fold_idx, score in enumerate(cv_scores):
            trial.report(score, fold_idx)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return mean CV score
        return float(cv_scores.mean())

    def search(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            Dictionary containing:
                - best_params: Best hyperparameters found
                - best_score: Best cross-validation score
                - best_trial: Best trial number
                - n_trials: Total number of trials run
                - n_complete: Number of completed trials
                - n_pruned: Number of pruned trials
                - study: Optuna study object

        Raises:
            ValueError: If X and y have incompatible shapes

        Example:
            >>> optimizer = OptunaOptimizer(model, search_space, n_trials=50)
            >>> results = optimizer.search(X_train, y_train)
            >>> print(f"Best R²: {results['best_score']:.4f}")
            Best R²: 0.8012
        """
        # Validate inputs
        self.validate_inputs(X, y)

        # Store data for objective function
        self.X = X
        self.y = y

        logger.info(f"Creating Optuna study: {self.study_name}")

        try:
            # Create or load study
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True,
            )

            # Optimize
            logger.info(f"Starting optimization: {self.n_trials} trials")
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=(self.verbose > 0),
            )

            # Count trial states
            n_complete = len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            )
            n_pruned = len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
            )

            # Store results
            self._best_params = self.study.best_params
            self._best_score = self.study.best_value
            self._search_results = {
                "best_params": self.study.best_params,
                "best_score": self.study.best_value,
                "best_trial": self.study.best_trial.number,
                "n_trials": len(self.study.trials),
                "n_complete": n_complete,
                "n_pruned": n_pruned,
                "random_state": self.random_state,
                "study": self.study,
            }

            logger.info(
                f"Optimization complete: best_score={self._best_score:.4f}, "
                f"complete={n_complete}, pruned={n_pruned}"
            )

            return self._search_results

        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            raise
