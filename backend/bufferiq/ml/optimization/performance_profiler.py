"""Track performance metrics during optimization."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class PerformanceProfiler:
    """
    Profile model performance during optimization.

    Tracks training time, memory usage, inference speed, and model size
    to enable multi-objective optimization and resource-aware tuning.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: list[dict[str, Any]] = []
        logger.info("Performance profiler initialized")

    def profile_training(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Profile model training performance.

        Args:
            model: Model to profile
            X: Training features
            y: Training targets

        Returns:
            Dictionary with performance metrics

        Example:
            >>> profiler = PerformanceProfiler()
            >>> profile = profiler.profile_training(model, X_train, y_train)
            >>> print(f"Training time: {profile['training_time']:.2f}s")
        """
        # Measure training time
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time

        # Measure inference speed
        start_time = time.time()
        _ = model.predict(X[:100])  # Sample 100 predictions
        inference_time = time.time() - start_time
        predictions_per_second = 100 / inference_time if inference_time > 0 else 0

        # Estimate model size
        model_size_mb = self._estimate_model_size(model)

        profile = {
            "training_time": training_time,
            "inference_time": inference_time,
            "predictions_per_second": predictions_per_second,
            "model_size_mb": model_size_mb,
        }

        self.profiles.append(profile)

        return profile

    def _estimate_model_size(self, model: BaseEstimator) -> float:
        """
        Estimate model size in MB.

        Args:
            model: Trained model

        Returns:
            Estimated size in MB
        """
        # For tree-based models, use number of trees/leaves
        if hasattr(model, "n_estimators"):
            # Rough estimate: 10KB per tree
            size_mb = model.n_estimators * 0.01
        elif hasattr(model, "tree_"):
            # Single decision tree
            size_mb = 0.01
        else:
            # Default estimate
            size_mb = 1.0

        return size_mb

    def visualize_performance(
        self,
        accuracy_scores: list[float],
        save_path: Path,
    ) -> None:
        """
        Visualize performance trade-offs.

        Args:
            accuracy_scores: List of accuracy scores corresponding to profiles
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        if len(self.profiles) == 0:
            logger.warning("No profiles to visualize")
            return

        if len(accuracy_scores) != len(self.profiles):
            raise ValueError(
                f"Length mismatch: {len(accuracy_scores)} scores vs "
                f"{len(self.profiles)} profiles"
            )

        # Extract metrics
        training_times = [p["training_time"] for p in self.profiles]
        model_sizes = [p["model_size_mb"] for p in self.profiles]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Accuracy vs Training Time
        axes[0].scatter(training_times, accuracy_scores, alpha=0.6)
        axes[0].set_xlabel("Training Time (s)")
        axes[0].set_ylabel("Accuracy (R²)")
        axes[0].set_title("Accuracy vs Training Time")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Accuracy vs Model Size
        axes[1].scatter(model_sizes, accuracy_scores, alpha=0.6, color="green")
        axes[1].set_xlabel("Model Size (MB)")
        axes[1].set_ylabel("Accuracy (R²)")
        axes[1].set_title("Accuracy vs Model Size")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance visualization saved to {save_path}")

    def export_profiles(self, save_path: Path) -> None:
        """
        Export performance profiles to JSON.

        Args:
            save_path: Path to save JSON file
        """
        import json

        data = {
            "n_profiles": len(self.profiles),
            "profiles": self.profiles,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance profiles exported to {save_path}")
