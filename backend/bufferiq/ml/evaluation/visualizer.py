"""Evaluation visualizations."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class EvaluationVisualizer:
    """Create evaluation visualizations."""

    def __init__(self, output_dir: str = "outputs/evaluations") -> None:
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for visualization outputs

        Example:
            >>> visualizer = EvaluationVisualizer()
            >>> visualizer.plot_residuals(y_true, y_pred, "residuals.png")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot residual analysis (2×2 grid).

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_residuals(y_true, y_pred, "residuals.png")
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual distribution
        axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Normal Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 1].scatter(y_pred, sqrt_abs_residuals, alpha=0.5, s=20)
        axes[1, 1].set_xlabel("Predicted Values")
        axes[1, 1].set_ylabel("√|Residuals|")
        axes[1, 1].set_title("Scale-Location Plot")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved residual plot to {save_path}")

        plt.close()

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot predicted vs actual scatter.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_predictions_vs_actual(y_true, y_pred, "pred_vs_actual.png")
        """
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors="k", linewidths=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        # Labels and title
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Predicted vs Actual (R² = {r2:.4f})", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Equal aspect ratio
        plt.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved predictions vs actual plot to {save_path}")

        plt.close()

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: int = 30,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot error distribution histogram.

        Args:
            y_true: True values
            y_pred: Predicted values
            bins: Number of histogram bins
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_error_distribution(y_true, y_pred, bins=30)
        """
        errors = y_true - y_pred

        plt.figure(figsize=(10, 6))

        plt.hist(errors, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero Error")
        plt.axvline(
            x=np.mean(errors),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean Error: {np.mean(errors):.4f}",
        )

        plt.xlabel("Error (Actual - Predicted)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Error Distribution", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved error distribution plot to {save_path}")

        plt.close()

    def plot_platform_performance(
        self, platform_metrics: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot platform-wise performance comparison.

        Args:
            platform_metrics: DataFrame with platform metrics
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_platform_performance(platform_metrics, "platforms.png")
        """
        plt.figure(figsize=(10, 6))

        # Bar chart of R² by platform
        platforms = platform_metrics["platform"].values
        r2_scores = platform_metrics["r2"].values

        plt.bar(platforms, r2_scores, color="steelblue", edgecolor="black")

        plt.xlabel("Platform", fontsize=12)
        plt.ylabel("R² Score", fontsize=12)
        plt.title("Model Performance by Platform", fontsize=14)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (platform, score) in enumerate(zip(platforms, r2_scores)):
            plt.text(i, score + 0.02, f"{score:.3f}", ha="center", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved platform performance plot to {save_path}")

        plt.close()

    def plot_temporal_performance(
        self,
        temporal_metrics: pd.DataFrame,
        metric: str = "r2",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot performance over time.

        Args:
            temporal_metrics: DataFrame with temporal metrics
            metric: Metric to plot
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_temporal_performance(temporal_metrics, "r2", "temporal.png")
        """
        plt.figure(figsize=(12, 6))

        periods = temporal_metrics["period"].values
        values = temporal_metrics[metric].values

        plt.plot(periods, values, marker="o", linewidth=2, markersize=8)

        plt.xlabel("Time Period", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f"Model Performance Over Time ({metric.upper()})", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved temporal performance plot to {save_path}")

        plt.close()

    def plot_learning_curve(
        self,
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot learning curve.

        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            save_path: Path to save plot

        Example:
            >>> visualizer.plot_learning_curve(
            ...     train_sizes, train_scores, val_scores, "learning_curve.png"
            ... )
        """
        plt.figure(figsize=(10, 6))

        plt.plot(train_sizes, train_scores, "o-", label="Training Score", linewidth=2)
        plt.plot(
            train_sizes, val_scores, "o-", label="Validation Score", linewidth=2
        )

        plt.xlabel("Training Set Size", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Learning Curve", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved learning curve plot to {save_path}")

        plt.close()