"""Tests for experiment tracker."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bufferiq.ml.training.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Test experiment tracker."""

    @pytest.fixture
    def temp_dir(self) -> str:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def tracker(self, temp_dir: str) -> ExperimentTracker:
        """Create experiment tracker."""
        return ExperimentTracker(experiment_name="test_experiment", base_dir=temp_dir)

    def test_init(self, tracker: ExperimentTracker) -> None:
        """Test initialization."""
        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_dir.exists()

    def test_log_params(self, tracker: ExperimentTracker) -> None:
        """Test logging parameters."""
        params = {"learning_rate": 0.01, "n_estimators": 100}
        tracker.log_params(params)

        assert tracker.get_params() == params

    def test_log_metric(self, tracker: ExperimentTracker) -> None:
        """Test logging single metric."""
        tracker.log_metric("accuracy", 0.95)

        metrics = tracker.get_metrics()
        assert "accuracy" in metrics
        assert len(metrics["accuracy"]) == 1
        assert metrics["accuracy"][0]["value"] == 0.95

    def test_log_metrics(self, tracker: ExperimentTracker) -> None:
        """Test logging multiple metrics."""
        metrics = {"mae": 2.5, "rmse": 3.2, "r2": 0.85}
        tracker.log_metrics(metrics)

        logged_metrics = tracker.get_metrics()
        assert "mae" in logged_metrics
        assert "rmse" in logged_metrics
        assert "r2" in logged_metrics

    def test_log_metric_with_step(self, tracker: ExperimentTracker) -> None:
        """Test logging metric with step."""
        tracker.log_metric("loss", 1.0, step=1)
        tracker.log_metric("loss", 0.5, step=2)

        metrics = tracker.get_metrics()
        assert len(metrics["loss"]) == 2
        assert metrics["loss"][0]["step"] == 1
        assert metrics["loss"][1]["step"] == 2

    def test_log_dataframe(self, tracker: ExperimentTracker) -> None:
        """Test logging DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tracker.log_dataframe(df, "test_data")

        csv_path = tracker.experiment_dir / "test_data.csv"
        assert csv_path.exists()

        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_experiment(self, tracker: ExperimentTracker) -> None:
        """Test saving experiment."""
        tracker.log_params({"param1": 1})
        tracker.log_metric("metric1", 0.5)

        exp_dir = tracker.save_experiment()

        # Check files exist
        assert Path(exp_dir).exists()
        assert (Path(exp_dir) / "config.json").exists()
        assert (Path(exp_dir) / "metrics.json").exists()
        assert (Path(exp_dir) / "metadata.json").exists()

    def test_load_experiment(self, tracker: ExperimentTracker) -> None:
        """Test loading experiment."""
        tracker.log_params({"param1": 1})
        tracker.log_metric("metric1", 0.5)
        exp_dir = tracker.save_experiment()

        loaded = ExperimentTracker.load_experiment(exp_dir)

        assert loaded.experiment_name == tracker.experiment_name
        assert loaded.get_params() == tracker.get_params()

    def test_list_experiments(self, temp_dir: str) -> None:
        """Test listing experiments."""
        # Create multiple experiments
        tracker1 = ExperimentTracker("exp1", base_dir=temp_dir)
        tracker1.save_experiment()

        tracker2 = ExperimentTracker("exp2", base_dir=temp_dir)
        tracker2.save_experiment()

        experiments = ExperimentTracker.list_experiments(base_dir=temp_dir)

        assert len(experiments) == 2
        assert any(e["experiment_name"] == "exp1" for e in experiments)
        assert any(e["experiment_name"] == "exp2" for e in experiments)
