"""Diversity analysis for ensemble models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class DiversityAnalyzer:
    """
    Analyze diversity of model predictions.

    Provides multiple diversity metrics to assess how different
    model predictions are from each other. Higher diversity
    generally leads to better ensemble performance.

    Example:
        >>> analyzer = DiversityAnalyzer()
        >>> predictions = np.column_stack([model1.predict(X), model2.predict(X)])
        >>> diversity = analyzer.correlation_diversity(predictions)
        >>> print(f"Diversity: {diversity:.4f}")
    """

    @staticmethod
    def correlation_diversity(predictions: np.ndarray) -> float:
        """
        Calculate correlation-based diversity.

        Diversity = 1 - average pairwise correlation
        Higher values indicate more diverse predictions.

        Args:
            predictions: Matrix of shape (n_samples, n_models)

        Returns:
            Diversity score (0-1, higher is more diverse)

        Example:
            >>> predictions = np.column_stack([pred1, pred2, pred3])
            >>> diversity = DiversityAnalyzer.correlation_diversity(predictions)
            >>> print(f"Correlation diversity: {diversity:.4f}")
        """
        if predictions.ndim != 2:
            raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

        if predictions.shape[1] < 2:
            raise ValueError("Need at least 2 models for diversity calculation")

        n_models = predictions.shape[1]
        correlations = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr, _ = pearsonr(predictions[:, i], predictions[:, j])
                correlations.append(corr)

        avg_correlation = np.mean(correlations)
        diversity = 1.0 - avg_correlation

        logger.info(
            f"Correlation diversity: {diversity:.4f} "
            f"(avg correlation: {avg_correlation:.4f})"
        )

        return diversity

    @staticmethod
    def disagreement_diversity(
        predictions: np.ndarray, threshold: float = 0.01
    ) -> float:
        """
        Calculate disagreement-based diversity.

        Measures how often models make different predictions
        (difference > threshold).

        Args:
            predictions: Matrix of shape (n_samples, n_models)
            threshold: Prediction difference threshold

        Returns:
            Disagreement rate (0-1, higher is more diverse)

        Example:
            >>> diversity = DiversityAnalyzer.disagreement_diversity(predictions)
            >>> print(f"Disagreement diversity: {diversity:.4f}")
        """
        if predictions.ndim != 2:
            raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

        if predictions.shape[1] < 2:
            raise ValueError("Need at least 2 models for diversity calculation")

        n_samples, n_models = predictions.shape
        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = np.abs(predictions[:, i] - predictions[:, j])
                disagreements += np.sum(diff > threshold)
                total_pairs += n_samples

        disagreement_rate = disagreements / total_pairs if total_pairs > 0 else 0.0

        logger.info(f"Disagreement diversity: {disagreement_rate:.4f}")

        return disagreement_rate

    @staticmethod
    def q_statistic(predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise Q-statistics.

        Q measures agreement between two models:
        Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)

        where Nij is count of samples where:
        - model i is correct (1) or incorrect (0)
        - model j is correct (1) or incorrect (0)

        Q values:
        - Q close to 1: models make same mistakes (low diversity)
        - Q close to -1: models make opposite mistakes (high diversity)
        - Q close to 0: models are independent

        Args:
            predictions: Matrix of shape (n_samples, n_models)
            y_true: True labels, shape (n_samples,)

        Returns:
            Q-statistic matrix, shape (n_models, n_models)

        Example:
            >>> q_matrix = DiversityAnalyzer.q_statistic(predictions, y_true)
            >>> print(f"Average Q-statistic: {np.mean(q_matrix):.4f}")
        """
        if predictions.ndim != 2:
            raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

        if len(predictions) != len(y_true):
            raise ValueError(
                f"predictions and y_true must have same length: "
                f"{len(predictions)} != {len(y_true)}"
            )

        n_models = predictions.shape[1]
        q_matrix = np.zeros((n_models, n_models))

        # Define correctness threshold (within 10% of true value)
        threshold = 0.1 * np.std(y_true)

        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Determine if each model is correct
                correct_i = np.abs(predictions[:, i] - y_true) < threshold
                correct_j = np.abs(predictions[:, j] - y_true) < threshold

                N11 = np.sum(correct_i & correct_j)
                N00 = np.sum(~correct_i & ~correct_j)
                N01 = np.sum(~correct_i & correct_j)
                N10 = np.sum(correct_i & ~correct_j)

                denominator = N11 * N00 + N01 * N10

                if denominator == 0:
                    q = 0.0
                else:
                    q = (N11 * N00 - N01 * N10) / denominator

                q_matrix[i, j] = q
                q_matrix[j, i] = q

        avg_q = np.mean(q_matrix[np.triu_indices_from(q_matrix, k=1)])
        logger.info(f"Average Q-statistic: {avg_q:.4f}")

        return q_matrix

    @staticmethod
    def visualize_correlation_matrix(
        predictions: np.ndarray,
        model_names: list,
        save_path: Path,
    ) -> None:
        """
        Visualize correlation matrix as heatmap.

        Args:
            predictions: Matrix of shape (n_samples, n_models)
            model_names: List of model names
            save_path: Path to save visualization

        Example:
            >>> DiversityAnalyzer.visualize_correlation_matrix(
            ...     predictions,
            ...     ['XGBoost', 'LightGBM', 'RandomForest'],
            ...     Path('outputs/correlation_matrix.png')
            ... )
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(predictions.T)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=model_names,
            yticklabels=model_names,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Model Prediction Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation matrix saved to {save_path}")

    @staticmethod
    def visualize_disagreement_matrix(
        predictions: np.ndarray,
        model_names: list,
        save_path: Path,
        threshold: float = 0.01,
    ) -> None:
        """
        Visualize disagreement matrix as heatmap.

        Args:
            predictions: Matrix of shape (n_samples, n_models)
            model_names: List of model names
            save_path: Path to save visualization
            threshold: Disagreement threshold

        Example:
            >>> DiversityAnalyzer.visualize_disagreement_matrix(
            ...     predictions,
            ...     ['XGBoost', 'LightGBM', 'RandomForest'],
            ...     Path('outputs/disagreement_matrix.png')
            ... )
        """
        n_samples, n_models = predictions.shape
        disagreement_matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    diff = np.abs(predictions[:, i] - predictions[:, j])
                    disagreement_rate = np.sum(diff > threshold) / n_samples
                    disagreement_matrix[i, j] = disagreement_rate

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            disagreement_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            xticklabels=model_names,
            yticklabels=model_names,
            square=True,
            cbar_kws={"label": "Disagreement Rate"},
        )
        plt.title(
            "Model Prediction Disagreement Matrix", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Disagreement matrix saved to {save_path}")

    @staticmethod
    def analyze_all(
        predictions: np.ndarray,
        y_true: np.ndarray,
        model_names: list,
        output_dir: Path,
    ) -> dict[str, float]:
        """
        Perform comprehensive diversity analysis.

        Args:
            predictions: Matrix of shape (n_samples, n_models)
            y_true: True labels
            model_names: List of model names
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with all diversity metrics

        Example:
            >>> metrics = DiversityAnalyzer.analyze_all(
            ...     predictions, y_true, model_names, Path('outputs/diversity')
            ... )
            >>> print(metrics)
        """
        logger.info("Starting comprehensive diversity analysis")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        corr_diversity = DiversityAnalyzer.correlation_diversity(predictions)
        disagree_diversity = DiversityAnalyzer.disagreement_diversity(predictions)
        q_matrix = DiversityAnalyzer.q_statistic(predictions, y_true)
        avg_q = np.mean(q_matrix[np.triu_indices_from(q_matrix, k=1)])

        # Create visualizations
        DiversityAnalyzer.visualize_correlation_matrix(
            predictions, model_names, output_dir / "correlation_matrix.png"
        )

        DiversityAnalyzer.visualize_disagreement_matrix(
            predictions, model_names, output_dir / "disagreement_matrix.png"
        )

        metrics = {
            "correlation_diversity": corr_diversity,
            "disagreement_diversity": disagree_diversity,
            "avg_q_statistic": avg_q,
        }

        logger.info(f"Diversity analysis complete: {metrics}")

        return metrics
