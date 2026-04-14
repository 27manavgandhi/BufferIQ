"""Cross-validation for model evaluation."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class CrossValidator:
    """Cross-validation for model evaluation."""

    def __init__(
        self,
        n_splits: int = 5,
        strategy: Literal["kfold", "timeseries"] = "timeseries",
        shuffle: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Initialize cross-validator.

        Args:
            n_splits: Number of folds
            strategy: 'kfold' or 'timeseries' (for temporal data)
            shuffle: Shuffle data before split (ignored for timeseries)
            random_state: Random seed

        Example:
            >>> cv = CrossValidator(n_splits=5, strategy='timeseries')
            >>> results = cv.cross_validate(trainer, X, y)
            >>> print(f"Mean R²: {results['mean_metrics']['r2']:.3f}")
        """
        self.n_splits = n_splits
        self.strategy = strategy
        self.shuffle = shuffle
        self.random_state = random_state

        # Create splitter
        if strategy == "timeseries":
            self.splitter = TimeSeriesSplit(n_splits=n_splits)
        else:
            self.splitter = KFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )

        # Storage for results
        self.fold_results_: list[dict[str, Any]] = []

    def cross_validate(
        self,
        trainer: BaseTrainer,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: list[str] = ["mae", "rmse", "r2"],
    ) -> dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            trainer: Trainer instance (will be cloned for each fold)
            X: Feature DataFrame
            y: Target Series
            metrics: List of metrics to compute

        Returns:
            Dict with:
                - fold_metrics: List of metric dicts per fold
                - mean_metrics: Mean across folds
                - std_metrics: Std across folds
                - cv_score: Primary metric mean

        Example:
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> trainer = SomeTrainer(model_name="rf")
            >>> cv = CrossValidator(n_splits=5)
            >>> results = cv.cross_validate(trainer, X, y)
        """
        logger.info(
            f"Starting {self.n_splits}-fold cross-validation using {self.strategy}"
        )

        fold_results: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.splitter.split(X), start=1
        ):
            logger.info(f"Training fold {fold_idx}/{self.n_splits}")

            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model (create fresh instance for each fold)
            trainer_fold = trainer.__class__(
                model_name=trainer.model_name,
                random_state=trainer.random_state,
                verbose=False,
            )

            # Build model with same hyperparameters
            if hasattr(trainer, "hyperparameters"):
                trainer_fold.build_model(trainer.hyperparameters)  # type: ignore
            else:
                trainer_fold.build_model({})

            # Train
            trainer_fold.train(X_train_fold, y_train_fold)

            # Evaluate
            fold_metrics = trainer_fold.evaluate(X_val_fold, y_val_fold)
            fold_results.append(fold_metrics)

            logger.info(f"Fold {fold_idx} results: {fold_metrics}")

        # Store fold results
        self.fold_results_ = fold_results

        # Calculate mean and std
        mean_metrics: dict[str, float] = {}
        std_metrics: dict[str, float] = {}

        for metric_name in fold_results[0].keys():
            values = [fold[metric_name] for fold in fold_results]
            mean_metrics[metric_name] = float(np.mean(values))
            std_metrics[metric_name] = float(np.std(values))

        # Primary metric is first in list
        primary_metric = list(mean_metrics.keys())[0]
        cv_score = mean_metrics[primary_metric]

        logger.info(f"Cross-validation complete. Mean metrics: {mean_metrics}")

        return {
            "fold_metrics": fold_results,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "cv_score": cv_score,
            "n_splits": self.n_splits,
            "strategy": self.strategy,
        }

    def get_cv_summary(self) -> pd.DataFrame:
        """
        Return cross-validation results as DataFrame.

        Returns:
            DataFrame with fold results

        Raises:
            ValueError: If cross_validate not called yet
        """
        if not self.fold_results_:
            raise ValueError("No cross-validation results available")

        df = pd.DataFrame(self.fold_results_)
        df.insert(0, "fold", range(1, len(df) + 1))

        # Add mean and std rows
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["fold"] = "mean"

        std_row = df.std(numeric_only=True).to_dict()
        std_row["fold"] = "std"

        summary_df = pd.concat(
            [df, pd.DataFrame([mean_row, std_row])], ignore_index=True
        )

        return summary_df
