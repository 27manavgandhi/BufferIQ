"""Tests for feature importance analyzer."""

import pandas as pd
import pytest
from unittest.mock import patch

from bufferiq.ml.evaluation.feature_importance import FeatureImportanceAnalyzer
from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer


class TestFeatureImportanceAnalyzer:
    """Test feature importance analyzer."""

    @pytest.fixture
    def trained_model(self) -> XGBoostTrainer:
        """Create trained model."""
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
        trainer.train(X, y)

        return trainer

    @pytest.fixture
    def analyzer(self, tmp_path: str) -> FeatureImportanceAnalyzer:
        """Create analyzer."""
        return FeatureImportanceAnalyzer(output_dir=str(tmp_path))

    def test_init(self, analyzer: FeatureImportanceAnalyzer) -> None:
        """Test initialization."""
        assert analyzer.output_dir.exists()

    def test_get_builtin_importance(
        self, analyzer: FeatureImportanceAnalyzer, trained_model: XGBoostTrainer
    ) -> None:
        """Test built-in importance extraction."""
        importance = analyzer.get_builtin_importance(trained_model)

        assert len(importance) == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert "rank" in importance.columns

    def test_calculate_permutation_importance(
        self, analyzer: FeatureImportanceAnalyzer, trained_model: XGBoostTrainer
    ) -> None:
        """Test permutation importance."""
        X = pd.DataFrame(
            {
                "f1": range(50),
                "f2": range(50, 100),
                "f3": range(100, 150),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(50)])

        perm_importance = analyzer.calculate_permutation_importance(
            trained_model, X, y, n_repeats=3
        )

        assert len(perm_importance) == 3
        assert "feature" in perm_importance.columns
        assert "importance_mean" in perm_importance.columns
        assert "importance_std" in perm_importance.columns

    def test_compare_importance_methods(
        self, analyzer: FeatureImportanceAnalyzer, trained_model: XGBoostTrainer
    ) -> None:
        """Test importance method comparison."""
        X = pd.DataFrame(
            {
                "f1": range(50),
                "f2": range(50, 100),
                "f3": range(100, 150),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(50)])

        comparison = analyzer.compare_importance_methods(trained_model, X, y, top_n=3)

        assert len(comparison) <= 3
        assert "feature" in comparison.columns
        assert "builtin_rank" in comparison.columns
        assert "permutation_rank" in comparison.columns

    @patch("matplotlib.pyplot.savefig")
    def test_plot_importance(
        self,
        mock_savefig,
        analyzer: FeatureImportanceAnalyzer,
        trained_model: XGBoostTrainer,
        tmp_path: str,
    ) -> None:
        """Test importance plotting."""
        importance = analyzer.get_builtin_importance(trained_model)

        save_path = str(tmp_path / "importance.png")
        analyzer.plot_importance(importance, top_n=3, save_path=save_path)

        # Verify savefig was called
        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_importance_comparison(
        self,
        mock_savefig,
        analyzer: FeatureImportanceAnalyzer,
        trained_model: XGBoostTrainer,
        tmp_path: str,
    ) -> None:
        """Test comparison plotting."""
        X = pd.DataFrame(
            {
                "f1": range(50),
                "f2": range(50, 100),
                "f3": range(100, 150),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(50)])

        comparison = analyzer.compare_importance_methods(trained_model, X, y)

        save_path = str(tmp_path / "comparison.png")
        analyzer.plot_importance_comparison(comparison, save_path=save_path)

        # Verify savefig was called
        assert mock_savefig.called
