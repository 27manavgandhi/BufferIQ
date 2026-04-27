"""Advanced visualizations for Optuna optimization."""

from pathlib import Path
from typing import Optional

import optuna

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class AdvancedOptimizationVisualizer:
    """
    Create advanced visualizations for Optuna studies.

    Provides optimization history, parameter importance, Pareto fronts,
    and other insightful plots.
    """

    def __init__(self, study: optuna.Study):
        """
        Initialize visualizer.

        Args:
            study: Completed Optuna study

        Example:
            >>> visualizer = AdvancedOptimizationVisualizer(study)
            >>> visualizer.plot_optimization_history('history.html')
        """
        self.study = study
        logger.info(f"Visualizer initialized for study: {study.study_name}")

    def plot_optimization_history(self, save_path: Path) -> None:
        """
        Plot optimization history (score vs trial number).

        Args:
            save_path: Path to save HTML file
        """
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_html(str(save_path))
        logger.info(f"Optimization history saved to {save_path}")

    def plot_param_importances(
        self,
        save_path: Path,
        target: Optional[int] = None,
    ) -> None:
        """
        Plot parameter importances.

        Args:
            save_path: Path to save HTML file
            target: Target objective index (for multi-objective)
        """
        try:
            fig = optuna.visualization.plot_param_importances(
                self.study,
                target=target,
            )
            fig.write_html(str(save_path))
            logger.info(f"Parameter importances saved to {save_path}")
        except Exception as e:
            logger.warning(f"Could not plot importances: {e}")

    def plot_parallel_coordinate(
        self,
        save_path: Path,
        params: Optional[list[str]] = None,
    ) -> None:
        """
        Plot parallel coordinate plot.

        Args:
            save_path: Path to save HTML file
            params: List of parameters to include (None for all)
        """
        fig = optuna.visualization.plot_parallel_coordinate(
            self.study,
            params=params,
        )
        fig.write_html(str(save_path))
        logger.info(f"Parallel coordinate plot saved to {save_path}")

    def plot_contour(
        self,
        save_path: Path,
        params: Optional[list[str]] = None,
    ) -> None:
        """
        Plot contour plot for parameter interactions.

        Args:
            save_path: Path to save HTML file
            params: List of 2 parameters to plot (None for auto-select)
        """
        try:
            fig = optuna.visualization.plot_contour(
                self.study,
                params=params,
            )
            fig.write_html(str(save_path))
            logger.info(f"Contour plot saved to {save_path}")
        except Exception as e:
            logger.warning(f"Could not plot contour: {e}")

    def plot_slice(
        self,
        save_path: Path,
        params: Optional[list[str]] = None,
    ) -> None:
        """
        Plot slice plot showing parameter effects.

        Args:
            save_path: Path to save HTML file
            params: List of parameters to include (None for all)
        """
        fig = optuna.visualization.plot_slice(
            self.study,
            params=params,
        )
        fig.write_html(str(save_path))
        logger.info(f"Slice plot saved to {save_path}")

    def plot_edf(self, save_path: Path) -> None:
        """
        Plot empirical distribution function.

        Args:
            save_path: Path to save HTML file
        """
        fig = optuna.visualization.plot_edf(self.study)
        fig.write_html(str(save_path))
        logger.info(f"EDF plot saved to {save_path}")

    def plot_timeline(self, save_path: Path) -> None:
        """
        Plot timeline of trials.

        Args:
            save_path: Path to save HTML file
        """
        fig = optuna.visualization.plot_timeline(self.study)
        fig.write_html(str(save_path))
        logger.info(f"Timeline plot saved to {save_path}")

    def create_all_visualizations(self, output_dir: Path) -> None:
        """
        Create all available visualizations.

        Args:
            output_dir: Directory to save all plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_optimization_history(output_dir / "optimization_history.html")
        self.plot_param_importances(output_dir / "param_importances.html")
        self.plot_parallel_coordinate(output_dir / "parallel_coordinate.html")
        self.plot_contour(output_dir / "contour.html")
        self.plot_slice(output_dir / "slice.html")
        self.plot_edf(output_dir / "edf.html")
        self.plot_timeline(output_dir / "timeline.html")

        logger.info(f"All visualizations saved to {output_dir}")
