"""Visualization utilities for data analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

# Set default style
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


class Visualizer:
    """Create professional visualizations for analysis."""

    def __init__(self, output_dir: str = "outputs/figures") -> None:
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_distribution(
        self,
        data: pd.Series,
        title: str,
        xlabel: Optional[str] = None,
        bins: int = 50,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot distribution with histogram and KDE.

        Args:
            data: Series to plot
            title: Plot title
            xlabel: X-axis label (default: series name)
            bins: Number of histogram bins
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> fig = viz.plot_distribution(df['engagement_rate'], 'Engagement Rate Distribution')
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram with KDE
        data.hist(bins=bins, alpha=0.6, ax=ax, density=True, label="Histogram")
        data.plot.kde(ax=ax, linewidth=2, label="KDE")

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.4f}",
        )

        ax.set_xlabel(xlabel or data.name or "Value")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved distribution plot", path=str(full_path))

        return fig

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        figsize: tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            df: DataFrame with numeric columns
            figsize: Figure size (width, height)
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_correlation_matrix(df, save_path='correlation.png')
        """
        # Calculate correlation matrix
        corr = df.corr()

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved correlation matrix", path=str(full_path))

        return fig

    def plot_time_series(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        hue: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot time series line chart.

        Args:
            df: DataFrame with time series data
            x: Column name for x-axis (time)
            y: Column name for y-axis (metric)
            title: Plot title
            hue: Column name for grouping (optional)
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_time_series(df, 'date', 'engagement_rate', 'Engagement Over Time')
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if hue:
            for group in df[hue].unique():
                group_data = df[df[hue] == group]
                ax.plot(
                    group_data[x],
                    group_data[y],
                    marker="o",
                    label=group,
                    linewidth=2,
                )
        else:
            ax.plot(df[x], df[y], marker="o", linewidth=2)

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if hue:
            ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved time series plot", path=str(full_path))

        return fig

    def plot_hourly_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = "engagement_rate",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot 7x24 heatmap of engagement by day and hour.

        Args:
            df: DataFrame with 'day_of_week', 'hour', and metric columns
            metric: Metric to visualize
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_hourly_heatmap(df, 'engagement_rate')
        """
        # Create pivot table
        pivot = df.pivot_table(
            values=metric, index="day_of_week", columns="hour", aggfunc="mean"
        )

        # Reorder days (Monday=0 to Sunday=6)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot.index = [day_names[int(i)] for i in pivot.index]

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": metric},
            ax=ax,
        )

        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Day of Week", fontsize=12)
        ax.set_title(
            f"Engagement Heatmap: {metric} by Day and Hour",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved hourly heatmap", path=str(full_path))

        return fig

    def plot_platform_comparison(
        self,
        df: pd.DataFrame,
        metric: str,
        plot_type: str = "box",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot platform comparison using box plots or violin plots.

        Args:
            df: DataFrame with 'platform' and metric columns
            metric: Metric to compare
            plot_type: Type of plot ('box' or 'violin')
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_platform_comparison(df, 'engagement_rate', 'violin')
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "box":
            sns.boxplot(data=df, x="platform", y=metric, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=df, x="platform", y=metric, ax=ax)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        ax.set_xlabel("Platform", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} by Platform", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved platform comparison", path=str(full_path))

        return fig

    def plot_scatter_with_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        hue: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot scatter plot with regression line.

        Args:
            df: DataFrame with x and y columns
            x: Column name for x-axis
            y: Column name for y-axis
            title: Plot title
            hue: Column name for color grouping (optional)
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_scatter_with_regression(df, 'content_length', 'engagement_rate', 'Length vs Engagement')
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot with regression line
        sns.regplot(
            data=df,
            x=x,
            y=y,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red", "linewidth": 2},
            ax=ax,
        )

        # Add color grouping if specified
        if hue:
            for group in df[hue].unique():
                group_data = df[df[hue] == group]
                ax.scatter(
                    group_data[x],
                    group_data[y],
                    alpha=0.6,
                    label=group,
                    s=50,
                )
            ax.legend()

        # Calculate and display R²
        from scipy.stats import pearsonr

        corr, p_value = pearsonr(df[x].dropna(), df[y].dropna())
        r_squared = corr**2

        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.4f}\np = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved scatter plot", path=str(full_path))

        return fig

    def plot_bar_chart(
        self,
        data: pd.Series,
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot bar chart.

        Args:
            data: Series with index as labels and values as heights
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure (relative to output_dir)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = viz.plot_bar_chart(platform_stats, 'Posts by Platform')
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        data.plot(kind="bar", ax=ax, color=sns.color_palette("deep"))

        ax.set_xlabel(xlabel or "Category", fontsize=12)
        ax.set_ylabel(ylabel or "Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, bbox_inches="tight")
            logger.info("Saved bar chart", path=str(full_path))

        return fig
