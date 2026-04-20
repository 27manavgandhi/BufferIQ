"""Model comparison utilities."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class ModelComparator:
    """Compare multiple trained models."""

    def __init__(self, output_dir: str = "outputs/evaluations/comparisons") -> None:
        """
        Initialize comparator.

        Args:
            output_dir: Directory for comparison outputs

        Example:
            >>> comparator = ModelComparator()
            >>> comparison = comparator.compare_metrics(models, X_test, y_test)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_metrics(
        self,
        models: dict[str, BaseTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare models on all metrics.

        Args:
            models: Dict of model_name -> trainer
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with model comparison

        Example:
            >>> models = {"XGBoost": xgb_trainer, "LightGBM": lgb_trainer}
            >>> comparison = comparator.compare_metrics(models, X_test, y_test)
        """
        results = []

        for name, trainer in models.items():
            logger.info(f"Evaluating {name}...")

            # Get metrics
            metrics = trainer.evaluate(X_test, y_test)
            metrics["model"] = name

            results.append(metrics)

        # Create DataFrame
        comparison_df = pd.DataFrame(results)

        # Reorder columns
        cols = ["model", "mae", "rmse", "r2", "mape"]
        cols = [c for c in cols if c in comparison_df.columns]
        comparison_df = comparison_df[
            cols + [c for c in comparison_df.columns if c not in cols]
        ]

        logger.info(f"Compared {len(models)} models")

        return comparison_df

    def compare_by_platform(
        self,
        models: dict[str, BaseTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        platforms: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare models per platform.

        Args:
            models: Dict of model_name -> trainer
            X_test: Test features
            y_test: Test target
            platforms: Platform labels

        Returns:
            Multi-index DataFrame

        Example:
            >>> platform_comparison = comparator.compare_by_platform(
            ...     models, X_test, y_test, platforms
            ... )
        """
        from bufferiq.ml.evaluation.evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        results = []

        for name, trainer in models.items():
            # Get predictions
            y_pred = trainer.predict(X_test)

            # Evaluate by platform
            platform_metrics = evaluator.evaluate_by_platform(y_test, y_pred, platforms)

            # Add model name
            platform_metrics["model"] = name

            results.append(platform_metrics)

        # Combine all results
        comparison_df = pd.concat(results, ignore_index=True)

        return comparison_df

    def statistical_comparison(
        self,
        models: dict[str, BaseTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Statistical significance testing between models.

        Args:
            models: Dict of model_name -> trainer
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with p-values

        Example:
            >>> sig_test = comparator.statistical_comparison(models, X_test, y_test)
        """
        # Get predictions from all models
        predictions = {}
        for name, trainer in models.items():
            predictions[name] = trainer.predict(X_test)

        # Calculate squared errors
        squared_errors = {}
        for name, y_pred in predictions.items():
            squared_errors[name] = (y_test.values - y_pred) ** 2

        # Pairwise t-tests
        model_names = list(models.keys())
        results = []

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                # Paired t-test on squared errors
                t_stat, p_value = stats.ttest_rel(
                    squared_errors[model1], squared_errors[model2]
                )

                results.append(
                    {
                        "model_1": model1,
                        "model_2": model2,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

        return pd.DataFrame(results)

    def plot_metric_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "r2",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot bar chart comparing models on single metric.

        Args:
            comparison_df: Comparison DataFrame
            metric: Metric to plot
            save_path: Path to save plot

        Example:
            >>> comparator.plot_metric_comparison(comparison, "r2", "comparison.png")
        """
        plt.figure(figsize=(10, 6))

        models = comparison_df["model"].values
        values = comparison_df[metric].values

        plt.bar(models, values, color="steelblue", edgecolor="black")

        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f"Model Comparison ({metric.upper()})", fontsize=14)
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (model, value) in enumerate(zip(models, values)):
            plt.text(i, value + 0.01, f"{value:.4f}", ha="center", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        plt.close()

    def plot_predictions_comparison(
        self,
        models: dict[str, BaseTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_size: int = 100,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot predictions from all models vs actual.

        Args:
            models: Dict of model_name -> trainer
            X_test: Test features
            y_test: Test target
            sample_size: Number of samples to plot
            save_path: Path to save plot

        Example:
            >>> comparator.plot_predictions_comparison(
            ...     models, X_test, y_test, sample_size=100
            ... )
        """
        # Sample data
        if len(X_test) > sample_size:
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[indices]
            y_sample = y_test.iloc[indices]
        else:
            X_sample = X_test
            y_sample = y_test

        plt.figure(figsize=(10, 8))

        # Plot actual values
        x_range = range(len(y_sample))

        # Get predictions from all models
        for name, trainer in models.items():
            y_pred = trainer.predict(X_sample)
            plt.scatter(x_range, y_pred, label=name, alpha=0.6, s=50)

        # Plot actual
        plt.scatter(
            x_range, y_sample.values, label="Actual", color="black", marker="x", s=100
        )

        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title("Predictions Comparison (Sample)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved predictions comparison plot to {save_path}")

        plt.close()

    def get_best_model(
        self,
        models: dict[str, BaseTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric: str = "r2",
    ) -> str:
        """
        Get best model based on metric.

        Args:
            models: Dict of model_name -> trainer
            X_test: Test features
            y_test: Test target
            metric: Metric to optimize

        Returns:
            Best model name

        Example:
            >>> best = comparator.get_best_model(models, X_test, y_test, "r2")
            >>> print(f"Best model: {best}")
        """
        comparison = self.compare_metrics(models, X_test, y_test)

        # Higher is better for r2, lower is better for mae/rmse
        if metric in ["r2", "explained_variance"]:
            best_idx = comparison[metric].idxmax()
        else:
            best_idx = comparison[metric].idxmin()

        best_model = comparison.loc[best_idx, "model"]

        logger.info(
            f"Best model: {best_model} ({metric}={comparison.loc[best_idx, metric]:.4f})"
        )

        return str(best_model)
