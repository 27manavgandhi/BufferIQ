"""Hyperparameter importance analysis for Optuna studies."""

from pathlib import Path
from typing import Optional

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class HyperparameterImportanceAnalyzer:
    """
    Analyze hyperparameter importance from completed Optuna study.

    Uses functional ANOVA (fANOVA) to determine which hyperparameters
    have the most impact on the objective function.
    """

    def __init__(self, study: optuna.Study):
        """
        Initialize importance analyzer.

        Args:
            study: Completed Optuna study

        Example:
            >>> analyzer = HyperparameterImportanceAnalyzer(study)
            >>> importance = analyzer.calculate_importance()
            >>> analyzer.visualize_importance(importance, 'importance.png')
        """
        self.study = study

        if len(study.trials) == 0:
            raise ValueError("Study has no trials. Cannot analyze importance.")

        logger.info(f"Importance analyzer initialized with {len(study.trials)} trials")

    def calculate_importance(
        self,
        method: str = "fanova",
        target: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Calculate hyperparameter importance.

        Args:
            method: Importance calculation method ('fanova')
            target: Target objective index (for multi-objective)

        Returns:
            Dictionary mapping parameter names to importance scores

        Example:
            >>> importance = analyzer.calculate_importance()
            >>> print(importance)
            {'learning_rate': 0.45, 'max_depth': 0.32, 'n_estimators': 0.15, ...}
        """
        if method == "fanova":
            evaluator = FanovaImportanceEvaluator(seed=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        try:
            importance = get_param_importances(
                self.study,
                evaluator=evaluator,
                target=target,
            )

            logger.info(f"Calculated importance for {len(importance)} parameters")
            return importance

        except Exception as e:
            logger.error(f"Importance calculation failed: {e}", exc_info=True)
            raise

    def visualize_importance(
        self,
        importance: dict[str, float],
        save_path: Path,
        top_n: int = 10,
    ) -> None:
        """
        Create bar chart of parameter importance.

        Args:
            importance: Dictionary of parameter importances
            save_path: Path to save visualization
            top_n: Number of top parameters to show
        """
        import matplotlib.pyplot as plt

        # Sort by importance
        sorted_params = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        if not sorted_params:
            logger.warning("No importance data to visualize")
            return

        params, scores = zip(*sorted_params)

        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(params, scores, color="steelblue")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Hyperparameter", fontsize=12)
        plt.title("Hyperparameter Importance", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Importance visualization saved to {save_path}")

    def export_rankings(self, importance: dict[str, float], save_path: Path) -> None:
        """
        Export importance rankings to JSON.

        Args:
            importance: Dictionary of parameter importances
            save_path: Path to save JSON file
        """
        import json

        # Sort by importance
        sorted_params = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create rankings
        rankings = {
            "rankings": [
                {
                    "rank": idx + 1,
                    "parameter": param,
                    "importance": float(score),
                }
                for idx, (param, score) in enumerate(sorted_params)
            ],
            "total_parameters": len(sorted_params),
        }

        # Save to JSON
        with open(save_path, "w") as f:
            json.dump(rankings, f, indent=2)

        logger.info(f"Importance rankings exported to {save_path}")
