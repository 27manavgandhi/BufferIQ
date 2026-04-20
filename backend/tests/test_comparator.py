"""Tests for model comparator."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from bufferiq.ml.evaluation.comparator import ModelComparator
from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer
from bufferiq.ml.trainers.lightgbm_trainer import LightGBMTrainer


class TestModelComparator:
    """Test model comparator."""

    @pytest.fixture
    def trained_models(self) -> dict:
        """Create trained models."""
        X = pd.DataFrame(
            {
                "f1": range(100),
                "f2": range(100, 200),
                "f3": range(200, 300),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(100)])

        # Train XGBoost
        xgb_trainer = XGBoostTrainer(verbose=False)
        xgb_trainer.build_model({"n_estimators": 10})
        xgb_trainer.train(X[:80], y[:80])

        # Train LightGBM
        lgb_trainer = LightGBMTrainer(verbose=False)
        lgb_trainer.build_model({"n_estimators": 10})
        lgb_trainer.train(X[:80], y[:80])

        return {"XGBoost": xgb_trainer, "LightGBM": lgb_trainer}

    @pytest.fixture
    def comparator(self, tmp_path: str) -> ModelComparator:
        """Create comparator."""
        return ModelComparator(output_dir=str(tmp_path))

    def test_init(self, comparator: ModelComparator) -> None:
        """Test initialization."""
        assert comparator.output_dir.exists()

    def test_compare_metrics(
        self, comparator: ModelComparator, trained_models: dict
    ) -> None:
        """Test metrics comparison."""
        X_test = pd.DataFrame(
            {
                "f1": range(100, 120),
                "f2": range(200, 220),
                "f3": range(300, 320),
            }
        )
        y_test = pd.Series([10 + i * 0.1 for i in range(100, 120)])

        comparison = comparator.compare_metrics(trained_models, X_test, y_test)

        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "mae" in comparison.columns
        assert "r2" in comparison.columns

    def test_compare_by_platform(
        self, comparator: ModelComparator, trained_models: dict
    ) -> None:
        """Test platform comparison."""
        X_test = pd.DataFrame(
            {
                "f1": range(100, 120),
                "f2": range(200, 220),
                "f3": range(300, 320),
            }
        )
        y_test = pd.Series([10 + i * 0.1 for i in range(100, 120)])
        platforms = pd.Series(["linkedin"] * 10 + ["twitter"] * 10)

        platform_comparison = comparator.compare_by_platform(
            trained_models, X_test, y_test, platforms
        )

        assert len(platform_comparison) > 0
        assert "model" in platform_comparison.columns
        assert "platform" in platform_comparison.columns

    def test_statistical_comparison(
        self, comparator: ModelComparator, trained_models: dict
    ) -> None:
        """Test statistical comparison."""
        X_test = pd.DataFrame(
            {
                "f1": range(100, 120),
                "f2": range(200, 220),
                "f3": range(300, 320),
            }
        )
        y_test = pd.Series([10 + i * 0.1 for i in range(100, 120)])

        sig_test = comparator.statistical_comparison(trained_models, X_test, y_test)

        assert len(sig_test) >= 1
        assert "model_1" in sig_test.columns
        assert "model_2" in sig_test.columns
        assert "p_value" in sig_test.columns

    def test_get_best_model(
        self, comparator: ModelComparator, trained_models: dict
    ) -> None:
        """Test best model selection."""
        X_test = pd.DataFrame(
            {
                "f1": range(100, 120),
                "f2": range(200, 220),
                "f3": range(300, 320),
            }
        )
        y_test = pd.Series([10 + i * 0.1 for i in range(100, 120)])

        best = comparator.get_best_model(trained_models, X_test, y_test, "r2")

        assert best in trained_models.keys()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_metric_comparison(
        self, mock_savefig, comparator: ModelComparator, tmp_path: str
    ) -> None:
        """Test metric comparison plotting."""
        comparison_df = pd.DataFrame(
            {
                "model": ["XGBoost", "LightGBM"],
                "r2": [0.75, 0.72],
                "mae": [0.15, 0.18],
            }
        )

        save_path = str(tmp_path / "comparison.png")
        comparator.plot_metric_comparison(comparison_df, "r2", save_path)

        assert mock_savefig.called
