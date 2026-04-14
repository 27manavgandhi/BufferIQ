"""Experiment tracking for ML training."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Track ML experiments."""

    def __init__(
        self, experiment_name: str, base_dir: str = "outputs/experiments"
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment
            base_dir: Base directory for experiments

        Creates directory structure:
            outputs/experiments/{experiment_name}/
                ├── config.json
                ├── metrics.json
                ├── model.joblib
                ├── feature_importance.csv
                ├── predictions.csv
                └── logs.txt
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name

        # Create directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking dicts
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: list[dict[str, str]] = []
        self.start_time = datetime.now()

        logger.info(f"Initialized experiment: {experiment_name}")

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Hyperparameters to log
        """
        self.params.update(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_metric(
        self, metric_name: str, value: float, step: Optional[int] = None
    ) -> None:
        """
        Log single metric.

        Args:
            metric_name: Name of metric
            value: Metric value
            step: Optional step/epoch number
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        metric_entry = {"value": float(value), "timestamp": datetime.now().isoformat()}

        if step is not None:
            metric_entry["step"] = step

        self.metrics[metric_name].append(metric_entry)

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dict of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

        logger.info(f"Logged {len(metrics)} metrics")

    def log_artifact(self, artifact_path: str, artifact_type: str) -> None:
        """
        Log artifact (model, plot, CSV, etc.).

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (model, plot, csv, etc.)
        """
        artifact_entry = {
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat(),
        }

        self.artifacts.append(artifact_entry)
        logger.info(f"Logged artifact: {artifact_path}")

    def log_model(self, model: Any, model_path: str) -> None:
        """
        Log trained model.

        Args:
            model: Model object to save
            model_path: Relative path within experiment dir
        """
        full_path = self.experiment_dir / model_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, full_path)
        self.log_artifact(str(full_path), "model")

        logger.info(f"Saved model to {full_path}")

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """
        Log DataFrame as CSV artifact.

        Args:
            df: DataFrame to save
            name: Name for the CSV file (without .csv extension)
        """
        csv_path = self.experiment_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        self.log_artifact(str(csv_path), "csv")

        logger.info(f"Saved DataFrame to {csv_path}")

    def get_metrics(self) -> dict[str, Any]:
        """
        Retrieve all logged metrics.

        Returns:
            Dict with all metrics
        """
        return self.metrics.copy()

    def get_params(self) -> dict[str, Any]:
        """
        Retrieve logged parameters.

        Returns:
            Dict with all parameters
        """
        return self.params.copy()

    def save_experiment(self) -> str:
        """
        Save experiment metadata.

        Returns:
            Path to experiment directory
        """
        # Save parameters
        params_path = self.experiment_dir / "config.json"
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=2)

        # Save metrics
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save artifacts list
        artifacts_path = self.experiment_dir / "artifacts.json"
        with open(artifacts_path, "w") as f:
            json.dump(self.artifacts, f, indent=2)

        # Save metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved experiment to {self.experiment_dir}")

        return str(self.experiment_dir)

    @classmethod
    def load_experiment(cls, experiment_dir: str) -> "ExperimentTracker":
        """
        Load experiment from directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Loaded ExperimentTracker instance

        Raises:
            FileNotFoundError: If experiment directory doesn't exist
        """
        exp_path = Path(experiment_dir)

        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        # Load metadata
        metadata_path = exp_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create tracker
        tracker = cls(
            experiment_name=metadata["experiment_name"],
            base_dir=str(exp_path.parent),
        )

        # Load params
        params_path = exp_path / "config.json"
        if params_path.exists():
            with open(params_path) as f:
                tracker.params = json.load(f)

        # Load metrics
        metrics_path = exp_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                tracker.metrics = json.load(f)

        # Load artifacts
        artifacts_path = exp_path / "artifacts.json"
        if artifacts_path.exists():
            with open(artifacts_path) as f:
                tracker.artifacts = json.load(f)

        logger.info(f"Loaded experiment from {experiment_dir}")

        return tracker

    @classmethod
    def list_experiments(
        cls, base_dir: str = "outputs/experiments"
    ) -> list[dict[str, Any]]:
        """
        List all experiments.

        Args:
            base_dir: Base directory to search

        Returns:
            List of dicts with experiment metadata
        """
        base_path = Path(base_dir)

        if not base_path.exists():
            return []

        experiments = []

        for exp_dir in base_path.iterdir():
            if not exp_dir.is_dir():
                continue

            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    metadata["experiment_dir"] = str(exp_dir)
                    experiments.append(metadata)

        # Sort by start time (most recent first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return experiments
