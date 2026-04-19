"""Tests for evaluation visualizer."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from bufferiq.ml.evaluation.visualizer import EvaluationVisualizer


class TestEvaluationVisualizer:
    """Test evaluation visualizer."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data."""
        np.random.seed(42)
        y_true = np.random.rand(100) * 10
        y_pred = y_true + np.random.randn(100) * 0.5
        return y_true, y_pred

    @pytest.fixture
    def visualizer(self, tmp_path: str) -> EvaluationVisualizer:
        """Create visualizer."""
        return EvaluationVisualizer(output_dir=str(tmp_path))

    def test_init(self, visualizer: EvaluationVisualizer) -> None:
        """Test initialization."""
        assert visualizer.output_dir.exists()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_residuals(
        self,
        mock_savefig,
        visualizer: EvaluationVisualizer,
        sample_data: tuple[np.ndarray, np.ndarray],
        tmp_path: str,
    ) -> None:
        """Test residual plotting."""
        y_true, y_pred = sample_data

        save_path = str(tmp_path / "residuals.png")
        visualizer.plot_residuals(y_true, y_pred, save_path)

        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_predictions_vs_actual(
        self,
        mock_savefig,
        visualizer: EvaluationVisualizer,
        sample_data: tuple[np.ndarray, np.ndarray],
        tmp_path: str,
    ) -> None:
        """Test predictions vs actual plotting."""
        y_true, y_pred = sample_data

        save_path = str(tmp_path / "pred_vs_actual.png")
        visualizer.plot_predictions_vs_actual(y_true, y_pred, save_path)

        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_error_distribution(
        self,
        mock_savefig,
        visualizer: EvaluationVisualizer,
        sample_data: tuple[np.ndarray, np.ndarray],
        tmp_path: str,
    ) -> None:
        """Test error distribution plotting."""
        y_true, y_pred = sample_data

        save_path = str(tmp_path / "error_dist.png")
        visualizer.plot_error_distribution(y_true, y_pred, save_path=save_path)

        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_platform_performance(
        self, mock_savefig, visualizer: EvaluationVisualizer, tmp_path: str
    ) -> None:
        """Test platform performance plotting."""
        platform_metrics = pd.DataFrame(
            {
                "platform": ["linkedin", "twitter", "bluesky"],
                "r2": [0.75, 0.70, 0.72],
                "mae": [0.15, 0.18, 0.16],
            }
        )

        save_path = str(tmp_path / "platforms.png")
        visualizer.plot_platform_performance(platform_metrics, save_path)

        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_temporal_performance(
        self, mock_savefig, visualizer: EvaluationVisualizer, tmp_path: str
    ) -> None:
        """Test temporal performance plotting."""
        temporal_metrics = pd.DataFrame(
            {
                "period": ["2024-01", "2024-02", "2024-03"],
                "r2": [0.70, 0.75, 0.72],
                "mae": [0.18, 0.15, 0.16],
            }
        )

        save_path = str(tmp_path / "temporal.png")
        visualizer.plot_temporal_performance(temporal_metrics, save_path=save_path)

        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    def test_plot_learning_curve(
        self, mock_savefig, visualizer: EvaluationVisualizer, tmp_path: str
    ) -> None:
        """Test learning curve plotting."""
        train_sizes = [50, 100, 150, 200]
        train_scores = [0.65, 0.75, 0.80, 0.85]
        val_scores = [0.60, 0.70, 0.72, 0.73]

        save_path = str(tmp_path / "learning_curve.png")
        visualizer.plot_learning_curve(
            train_sizes, train_scores, val_scores, save_path
        )

        assert mock_savefig.called