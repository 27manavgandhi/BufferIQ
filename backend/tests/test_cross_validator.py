"""Tests for cross-validator."""

import pandas as pd
import pytest

from bufferiq.ml.training.cross_validator import CrossValidator
from bufferiq.ml.training.trainer_base import BaseTrainer


class MockTrainer(BaseTrainer):
    """Mock trainer for testing."""

    def __init__(self, model_name: str = "mock", random_state: int = 42) -> None:
        super().__init__(model_name, random_state)
        self.hyperparameters = {}

    def build_model(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters
        self.model = "mock_model"

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None
    ) -> dict:
        return {"loss": 0.0}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        return {"mae": 1.0, "rmse": 1.5, "r2": 0.8}

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame()


class TestCrossValidator:
    """Test cross-validator."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data."""
        X = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "feature2": list(range(100, 200)),
            }
        )
        y = pd.Series(list(range(100)))
        return X, y

    @pytest.fixture
    def trainer(self) -> MockTrainer:
        """Create mock trainer."""
        return MockTrainer()

    def test_init_timeseries(self) -> None:
        """Test initialization with time series split."""
        cv = CrossValidator(n_splits=5, strategy="timeseries")

        assert cv.n_splits == 5
        assert cv.strategy == "timeseries"

    def test_init_kfold(self) -> None:
        """Test initialization with k-fold split."""
        cv = CrossValidator(n_splits=5, strategy="kfold", shuffle=True)

        assert cv.n_splits == 5
        assert cv.strategy == "kfold"
        assert cv.shuffle

    def test_cross_validate_basic(
        self, trainer: MockTrainer, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test basic cross-validation."""
        X, y = sample_data
        cv = CrossValidator(n_splits=3, strategy="kfold")

        results = cv.cross_validate(trainer, X, y)

        assert "fold_metrics" in results
        assert "mean_metrics" in results
        assert "std_metrics" in results
        assert "cv_score" in results
        assert len(results["fold_metrics"]) == 3

    def test_cross_validate_metrics(
        self, trainer: MockTrainer, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test cross-validation computes metrics."""
        X, y = sample_data
        cv = CrossValidator(n_splits=3, strategy="kfold")

        results = cv.cross_validate(trainer, X, y, metrics=["mae", "rmse", "r2"])

        assert "mae" in results["mean_metrics"]
        assert "rmse" in results["mean_metrics"]
        assert "r2" in results["mean_metrics"]

    def test_cross_validate_timeseries(
        self, trainer: MockTrainer, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test cross-validation with time series split."""
        X, y = sample_data
        cv = CrossValidator(n_splits=3, strategy="timeseries")

        results = cv.cross_validate(trainer, X, y)

        assert results["strategy"] == "timeseries"
        assert len(results["fold_metrics"]) == 3

    def test_get_cv_summary(
        self, trainer: MockTrainer, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test getting CV summary."""
        X, y = sample_data
        cv = CrossValidator(n_splits=3, strategy="kfold")

        cv.cross_validate(trainer, X, y)
        summary = cv.get_cv_summary()

        assert "fold" in summary.columns
        assert len(summary) == 5  # 3 folds + mean + std

    def test_get_cv_summary_not_run(self) -> None:
        """Test get summary before CV raises error."""
        cv = CrossValidator(n_splits=3)

        with pytest.raises(ValueError, match="No cross-validation results"):
            cv.get_cv_summary()
