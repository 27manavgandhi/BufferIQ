"""Feature importance analysis with multiple methods."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze feature importance with multiple methods."""

    def __init__(
        self, output_dir: str = "outputs/evaluations/feature_importance"
    ) -> None:
        """
        Initialize analyzer.

        Args:
            output_dir: Directory for importance plots

        Example:
            >>> analyzer = FeatureImportanceAnalyzer()
            >>> importance = analyzer.get_builtin_importance(trainer)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_builtin_importance(self, trainer: BaseTrainer) -> pd.DataFrame:
        """
        Get model's built-in feature importance.

        Args:
            trainer: Trained model

        Returns:
            DataFrame with feature importance

        Example:
            >>> importance = analyzer.get_builtin_importance(trainer)
            >>> print(importance.head())
        """
        importance_df = trainer.get_feature_importance()

        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)

        logger.info(f"Extracted built-in importance for {len(importance_df)} features")

        return importance_df

    def calculate_permutation_importance(
        self,
        trainer: BaseTrainer,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Calculate permutation importance.

        Args:
            trainer: Trained model
            X: Features
            y: Target
            n_repeats: Number of permutations
            random_state: Random seed

        Returns:
            DataFrame with permutation importance

        Example:
            >>> perm_imp = analyzer.calculate_permutation_importance(
            ...     trainer, X_test, y_test, n_repeats=10
            ... )
        """
        if trainer.model is None:
            raise ValueError("Model not trained")

        logger.info(f"Calculating permutation importance ({n_repeats} repeats)...")

        # Calculate permutation importance
        result = permutation_importance(
            trainer.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values(
            "importance_mean", ascending=False
        ).reset_index(drop=True)

        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)

        logger.info("Permutation importance calculated")

        return importance_df

    def compare_importance_methods(
        self, trainer: BaseTrainer, X: pd.DataFrame, y: pd.Series, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Compare rankings from different methods.

        Args:
            trainer: Trained model
            X: Features
            y: Target
            top_n: Top N features to compare

        Returns:
            DataFrame with method comparison

        Example:
            >>> comparison = analyzer.compare_importance_methods(
            ...     trainer, X_test, y_test, top_n=20
            ... )
        """
        # Get built-in importance
        builtin = self.get_builtin_importance(trainer)
        builtin = builtin.set_index("feature")["rank"].to_dict()

        # Get permutation importance
        permutation = self.calculate_permutation_importance(trainer, X, y, n_repeats=5)
        permutation = permutation.set_index("feature")["rank"].to_dict()

        # Combine
        all_features = set(builtin.keys()) | set(permutation.keys())

        comparison_data = []
        for feature in all_features:
            comparison_data.append(
                {
                    "feature": feature,
                    "builtin_rank": builtin.get(feature, len(builtin) + 1),
                    "permutation_rank": permutation.get(feature, len(permutation) + 1),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Calculate average rank
        comparison_df["avg_rank"] = (
            comparison_df["builtin_rank"] + comparison_df["permutation_rank"]
        ) / 2

        # Sort by average rank
        comparison_df = comparison_df.sort_values("avg_rank").reset_index(drop=True)

        return comparison_df.head(top_n)

    def plot_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot feature importance bar chart.

        Args:
            importance_df: Importance DataFrame
            top_n: Top N features to plot
            title: Plot title
            save_path: Path to save plot

        Example:
            >>> analyzer.plot_importance(
            ...     importance, top_n=20, save_path="importance.png"
            ... )
        """
        # Get top N features
        plot_data = importance_df.head(top_n).copy()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot horizontal bar chart
        plt.barh(
            range(len(plot_data)), plot_data["importance"].values, color="steelblue"
        )

        # Set labels
        plt.yticks(range(len(plot_data)), plot_data["feature"].values)
        plt.xlabel("Importance")
        plt.title(title)

        # Invert y-axis (highest importance at top)
        plt.gca().invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved importance plot to {save_path}")

        plt.close()

    def plot_importance_comparison(
        self,
        comparison_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot comparison of importance methods.

        Args:
            comparison_df: Comparison DataFrame
            top_n: Top N features to plot
            save_path: Path to save plot

        Example:
            >>> analyzer.plot_importance_comparison(
            ...     comparison, top_n=15, save_path="comparison.png"
            ... )
        """
        # Get top N features
        plot_data = comparison_df.head(top_n).copy()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(plot_data))
        width = 0.35

        # Plot bars
        ax.barh(
            x - width / 2,
            plot_data["builtin_rank"].values,
            width,
            label="Built-in",
            color="steelblue",
        )
        ax.barh(
            x + width / 2,
            plot_data["permutation_rank"].values,
            width,
            label="Permutation",
            color="coral",
        )

        # Set labels
        ax.set_yticks(x)
        ax.set_yticklabels(plot_data["feature"].values)
        ax.set_xlabel("Rank (lower is better)")
        ax.set_title("Feature Importance Method Comparison")
        ax.legend()

        # Invert y-axis
        ax.invert_yaxis()
        ax.invert_xaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        plt.close()
