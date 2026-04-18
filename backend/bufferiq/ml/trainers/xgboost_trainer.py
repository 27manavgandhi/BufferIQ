"""XGBoost trainer implementation."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class XGBoostTrainer(BaseTrainer):
    """XGBoost regression trainer."""

    def __init__(
        self, model_name: str = "xgboost", random_state: int = 42, verbose: bool = True
    ) -> None:
        """
        Initialize XGBoost trainer.

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress

        Example:
            >>> trainer = XGBoostTrainer(random_state=42)
            >>> hyperparams = {"n_estimators": 100, "max_depth": 6}
            >>> trainer.build_model(hyperparams)
        """
        super().__init__(model_name, random_state, verbose)
        self.hyperparameters: Dict[str, Any] = {}

    def build_model(self, hyperparameters: Dict[str, Any]) -> xgb.XGBRegressor:
        """
        Build XGBoost model with hyperparameters.

        Args:
            hyperparameters: Model hyperparameters

        Returns:
            Initialized XGBoost model

        Example:
            >>> trainer = XGBoostTrainer()
            >>> model = trainer.build_model({
            ...     "n_estimators": 100,
            ...     "max_depth": 6,
            ...     "learning_rate": 0.1
            ... })
        """
        self.hyperparameters = hyperparameters.copy()

        # Set default hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 1 if self.verbose else 0,
        }

        # Update with provided hyperparameters
        params.update(hyperparameters)

        # Ensure random_state is set
        params["random_state"] = self.random_state

        self.model = xgb.XGBRegressor(**params)

        if self.verbose:
            logger.info(f"Built XGBoost model with params: {params}")

        return self.model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Dict with training metrics

        Example:
            >>> trainer = XGBoostTrainer()
            >>> trainer.build_model({"n_estimators": 100})
            >>> metrics = trainer.train(X_train, y_train, X_val, y_val)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Store feature names
        self.feature_names_ = list(X_train.columns)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        if self.verbose:
            logger.info(f"Training XGBoost on {len(X_train)} samples")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=self.verbose,
        )

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_metrics = {
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "train_rmse": float(
                np.sqrt(mean_squared_error(y_train, y_train_pred))
            ),
            "train_r2": float(r2_score(y_train, y_train_pred)),
        }

        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_metrics = {
                "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                "val_r2": float(r2_score(y_val, y_val_pred)),
            }
            train_metrics.update(val_metrics)

        if self.verbose:
            logger.info(f"Training complete. Metrics: {train_metrics}")

        return train_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained

        Example:
            >>> predictions = trainer.predict(X_test)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dict with metrics (MAE, RMSE, R2)

        Example:
            >>> metrics = trainer.evaluate(X_test, y_test)
            >>> print(f"R²: {metrics['r2']:.4f}")
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.predict(X)

        metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
            "mape": float(np.mean(np.abs((y - y_pred) / y)) * 100)
            if (y != 0).all()
            else 0.0,
        }

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns: feature, importance

        Raises:
            ValueError: If model not trained

        Example:
            >>> importance = trainer.get_feature_importance()
            >>> print(importance.head())
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.feature_names_ is None:
            raise ValueError("Feature names not available")

        # Get feature importance
        importance_scores = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": importance_scores,
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        self.feature_importance_ = importance_df

        return importance_df