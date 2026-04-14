"""Abstract base class for model trainers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    def __init__(
        self, model_name: str, random_state: int = 42, verbose: bool = True
    ) -> None:
        """
        Initialize trainer.

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress

        Example:
            >>> class MyTrainer(BaseTrainer):
            ...     def build_model(self, hyperparameters):
            ...         return SomeModel(**hyperparameters)
        """
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose
        self.model: Optional[Any] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[list[str]] = None

    @abstractmethod
    def build_model(self, hyperparameters: dict[str, Any]) -> Any:
        """
        Build model with hyperparameters.

        Args:
            hyperparameters: Model hyperparameters

        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Train model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Dict with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dict with metrics (MAE, RMSE, R2, etc.)
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns: feature, importance

        Raises:
            ValueError: If model doesn't support feature importance
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Cannot save untrained model")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "random_state": self.random_state,
            "feature_names": self.feature_names_,
            "feature_importance": self.feature_importance_,
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load_model(cls, path: str) -> "BaseTrainer":
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Loaded trainer instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Create instance
        trainer = cls(
            model_name=model_data["model_name"],
            random_state=model_data["random_state"],
        )

        trainer.model = model_data["model"]
        trainer.feature_names_ = model_data.get("feature_names")
        trainer.feature_importance_ = model_data.get("feature_importance")

        logger.info(f"Loaded model from {path}")

        return trainer
