"""Tests for model evaluator."""

import numpy as np
import pandas as pd
import pytest

from bufferiq.ml.evaluation.evaluator import ModelEvaluator
from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer


class TestModelEvaluator:
    """Test model evaluator."""

    @pytest.fixture
    def sample_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample predictions."""
        np.random.seed(42)
        y_true = np.random.rand(100) * 10
        y_pred = y_true + np.random.randn(100) * 0.5
        return y_true, y_pred

    @pytest.fixture
    def evaluator(self, tmp_path: str) -> ModelEvaluator:
        """Create evaluator."""
        return ModelEvaluator(output_dir=str(tmp_path))

    def test_init(self, evaluator: ModelEvaluator) -> None:
        """Test initialization."""
        assert evaluator.output_dir.exists()
        assert (evaluator.output_dir / "residual_plots").exists()

    def test_calculate_metrics(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test metrics calculation."""
        y_true, y_pred = sample_predictions

        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "max_error" in metrics
        assert isinstance(metrics["mae"], float)
        assert metrics["r2"] >= 0

    def test_calculate_metrics_empty_arrays(self, evaluator: ModelEvaluator) -> None:
        """Test with empty arrays raises error."""
        with pytest.raises(ValueError, match="Empty input arrays"):
            evaluator.calculate_metrics(np.array([]), np.array([]))

    def test_calculate_metrics_different_lengths(
        self, evaluator: ModelEvaluator
    ) -> None:
        """Test with different length arrays raises error."""
        with pytest.raises(ValueError, match="same length"):
            evaluator.calculate_metrics(np.array([1, 2, 3]), np.array([1, 2]))

    def test_evaluate_by_platform(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test platform-wise evaluation."""
        y_true, y_pred = sample_predictions

        platforms = pd.Series(["linkedin"] * 40 + ["twitter"] * 30 + ["bluesky"] * 30)

        platform_metrics = evaluator.evaluate_by_platform(
            pd.Series(y_true), y_pred, platforms
        )

        assert len(platform_metrics) == 3
        assert "platform" in platform_metrics.columns
        assert "mae" in platform_metrics.columns
        assert set(platform_metrics["platform"]) == {"linkedin", "twitter", "bluesky"}

    def test_evaluate_by_time_period(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test temporal evaluation."""
        y_true, y_pred = sample_predictions

        timestamps = pd.date_range("2024-01-01", periods=100, freq="D")

        temporal_metrics = evaluator.evaluate_by_time_period(
            pd.Series(y_true), y_pred, pd.Series(timestamps), period="month"
        )

        assert len(temporal_metrics) > 0
        assert "period" in temporal_metrics.columns
        assert "mae" in temporal_metrics.columns

    def test_evaluate_by_content_type(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test content type evaluation."""
        y_true, y_pred = sample_predictions

        features = pd.DataFrame(
            {
                "has_url": [True] * 50 + [False] * 50,
                "hashtag_count": [0] * 30 + [2] * 70,
                "text_length": np.random.randint(50, 300, 100),
            }
        )

        content_metrics = evaluator.evaluate_by_content_type(
            pd.Series(y_true), y_pred, features
        )

        assert len(content_metrics) > 0
        assert "content_type" in content_metrics.columns

    def test_calculate_residuals(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test residual calculation."""
        y_true, y_pred = sample_predictions

        residuals = evaluator.calculate_residuals(y_true, y_pred)

        assert len(residuals) == len(y_true)
        assert isinstance(residuals, np.ndarray)

    def test_identify_worst_predictions(
        self,
        evaluator: ModelEvaluator,
        sample_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test worst predictions identification."""
        y_true, y_pred = sample_predictions

        features = pd.DataFrame(
            {
                "text_length": np.random.randint(50, 300, 100),
                "hashtag_count": np.random.randint(0, 5, 100),
            }
        )

        worst = evaluator.identify_worst_predictions(
            pd.Series(y_true), y_pred, features, top_n=10
        )

        assert len(worst) == 10
        assert "actual" in worst.columns
        assert "predicted" in worst.columns
        assert "error" in worst.columns


class TestModelEvaluatorIntegration:
    """Integration tests with real trainer."""

    @pytest.fixture
    def trained_model(self) -> XGBoostTrainer:
        """Create and train a simple model."""
        X = pd.DataFrame(
            {
                "f1": range(100),
                "f2": range(100, 200),
                "f3": range(200, 300),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(100)])

        trainer = XGBoostTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})
        trainer.train(X[:80], y[:80], X[80:], y[80:])

        return trainer

    def test_generate_evaluation_summary(
        self, trained_model: XGBoostTrainer, tmp_path: str
    ) -> None:
        """Test comprehensive evaluation summary."""
        X_test = pd.DataFrame(
            {
                "f1": range(100, 120),
                "f2": range(200, 220),
                "f3": range(300, 320),
            }
        )
        y_test = pd.Series([10 + i * 0.1 for i in range(100, 120)])
        platforms = pd.Series(["linkedin"] * 10 + ["twitter"] * 10)
        timestamps = pd.date_range("2024-01-01", periods=20)

        evaluator = ModelEvaluator(output_dir=str(tmp_path))

        summary = evaluator.generate_evaluation_summary(
            trained_model, X_test, y_test, platforms, pd.Series(timestamps)
        )

        assert "overall_metrics" in summary
        assert "platform_metrics" in summary
        assert "temporal_metrics" in summary
        assert summary["overall_metrics"]["r2"] > 0
