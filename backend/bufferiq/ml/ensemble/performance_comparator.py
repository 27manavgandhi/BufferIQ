"""Compare ensemble performance against base models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class EnsemblePerformanceComparator:
    """
    Compare ensemble performance against base models.

    Provides statistical tests and visualizations to demonstrate
    ensemble improvement over individual models.

    Example:
        >>> comparator = EnsemblePerformanceComparator()
        >>> results = comparator.compare(
        ...     ensemble, base_models, X_test, y_test
        ... )
        >>> comparator.visualize_comparison(results, 'comparison.png')
    """

    @staticmethod
    def compare(
        ensemble: BaseEstimator,
        base_models: list[BaseEstimator],
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_names: list[str],
    ) -> dict[str, any]:
        """
        Compare ensemble against base models.

        Args:
            ensemble: Fitted ensemble model
            base_models: List of fitted base models
            X_test: Test features
            y_test: Test targets
            model_names: Names of base models

        Returns:
            Dictionary with comparison results

        Example:
            >>> results = comparator.compare(
            ...     ensemble, [model1, model2], X_test, y_test,
            ...     ['XGBoost', 'LightGBM']
            ... )
        """
        logger.info("Comparing ensemble against base models")

        # Get ensemble predictions
        ensemble_pred = ensemble.predict(X_test)

        # Calculate ensemble metrics
        ensemble_metrics = {
            "r2": r2_score(y_test, ensemble_pred),
            "mae": mean_absolute_error(y_test, ensemble_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        }

        # Calculate base model metrics
        base_metrics = []
        base_predictions = []

        for i, model in enumerate(base_models):
            pred = model.predict(X_test)
            base_predictions.append(pred)

            metrics = {
                "name": model_names[i],
                "r2": r2_score(y_test, pred),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            }
            base_metrics.append(metrics)

            logger.debug(
                f"{model_names[i]}: R²={metrics['r2']:.4f}, "
                f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}"
            )

        logger.info(
            f"Ensemble: R²={ensemble_metrics['r2']:.4f}, "
            f"MAE={ensemble_metrics['mae']:.4f}, "
            f"RMSE={ensemble_metrics['rmse']:.4f}"
        )

        # Calculate improvements
        best_base_r2 = max(m["r2"] for m in base_metrics)
        improvement = (ensemble_metrics["r2"] - best_base_r2) / best_base_r2 * 100

        logger.info(f"Improvement over best base model: {improvement:.2f}%")

        # Statistical tests
        statistical_tests = EnsemblePerformanceComparator._statistical_tests(
            ensemble_pred, base_predictions, y_test
        )

        return {
            "ensemble_metrics": ensemble_metrics,
            "base_metrics": base_metrics,
            "improvement_pct": improvement,
            "statistical_tests": statistical_tests,
        }

    @staticmethod
    def _statistical_tests(
        ensemble_pred: np.ndarray,
        base_predictions: list[np.ndarray],
        y_test: np.ndarray,
    ) -> dict[str, any]:
        """
        Perform statistical significance tests.

        Args:
            ensemble_pred: Ensemble predictions
            base_predictions: List of base model predictions
            y_test: True targets

        Returns:
            Dictionary with test results
        """
        logger.info("Performing statistical significance tests")

        # Compute squared errors
        ensemble_errors = (ensemble_pred - y_test) ** 2

        tests = []
        for i, base_pred in enumerate(base_predictions):
            base_errors = (base_pred - y_test) ** 2

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(ensemble_errors, base_errors)

            # Wilcoxon signed-rank test (non-parametric alternative)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ensemble_errors, base_errors)

            tests.append(
                {
                    "model_index": i,
                    "paired_t_test": {
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    },
                    "wilcoxon_test": {
                        "statistic": float(wilcoxon_stat),
                        "p_value": float(wilcoxon_p),
                        "significant": wilcoxon_p < 0.05,
                    },
                }
            )

            logger.debug(
                f"Model {i}: t-test p={p_value:.4f}, " f"wilcoxon p={wilcoxon_p:.4f}"
            )

        return tests

    @staticmethod
    def visualize_comparison(
        results: dict[str, any],
        save_path: Path,
    ) -> None:
        """
        Visualize ensemble vs base model comparison.

        Args:
            results: Comparison results from compare()
            save_path: Path to save visualization

        Example:
            >>> comparator.visualize_comparison(
            ...     results, Path('outputs/comparison.png')
            ... )
        """
        ensemble_metrics = results["ensemble_metrics"]
        base_metrics = results["base_metrics"]

        # Prepare data
        model_names = [m["name"] for m in base_metrics] + ["Ensemble"]
        r2_scores = [m["r2"] for m in base_metrics] + [ensemble_metrics["r2"]]
        mae_scores = [m["mae"] for m in base_metrics] + [ensemble_metrics["mae"]]
        rmse_scores = [m["rmse"] for m in base_metrics] + [ensemble_metrics["rmse"]]

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Colors: base models in blue, ensemble in orange
        colors = ["steelblue"] * len(base_metrics) + ["coral"]

        # R² plot
        axes[0].bar(model_names, r2_scores, color=colors)
        axes[0].set_ylabel("R² Score", fontsize=12)
        axes[0].set_title("R² Comparison", fontsize=14, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].tick_params(axis="x", rotation=45)

        # MAE plot
        axes[1].bar(model_names, mae_scores, color=colors)
        axes[1].set_ylabel("MAE", fontsize=12)
        axes[1].set_title(
            "MAE Comparison (lower is better)", fontsize=14, fontweight="bold"
        )
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis="x", rotation=45)

        # RMSE plot
        axes[2].bar(model_names, rmse_scores, color=colors)
        axes[2].set_ylabel("RMSE", fontsize=12)
        axes[2].set_title(
            "RMSE Comparison (lower is better)", fontsize=14, fontweight="bold"
        )
        axes[2].grid(axis="y", alpha=0.3)
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Comparison visualization saved to {save_path}")

    @staticmethod
    def export_report(
        results: dict[str, any],
        save_path: Path,
    ) -> None:
        """
        Export comparison report to JSON.

        Args:
            results: Comparison results
            save_path: Path to save JSON report

        Example:
            >>> comparator.export_report(
            ...     results, Path('outputs/comparison_report.json')
            ... )
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comparison report exported to {save_path}")
