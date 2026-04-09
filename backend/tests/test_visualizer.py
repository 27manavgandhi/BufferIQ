"""Tests for visualizer."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from bufferiq.ml.analysis.visualizer import Visualizer


@pytest.fixture
def viz():
    """Create visualizer with temp output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Visualizer(output_dir=tmpdir)


@pytest.fixture
def sample_data():
    """Create sample data for visualization testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.05, 0.02, 100), name="engagement_rate")


@pytest.fixture
def sample_df():
    """Create sample DataFrame for visualization testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "engagement_rate": np.random.normal(0.05, 0.02, 100),
            "likes": np.random.poisson(20, 100),
            "comments": np.random.poisson(5, 100),
            "platform": np.random.choice(["linkedin", "twitter"], 100),
            "hour": np.random.randint(0, 24, 100),
            "day_of_week": np.random.randint(0, 7, 100),
            "content_length": np.random.randint(50, 500, 100),
        }
    )


def test_plot_distribution(viz, sample_data):
    """Test distribution plotting."""
    fig = viz.plot_distribution(sample_data, "Test Distribution")

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_distribution_with_save(viz, sample_data):
    """Test distribution plotting with save."""
    save_path = "test_dist.png"
    fig = viz.plot_distribution(sample_data, "Test Distribution", save_path=save_path)

    assert fig is not None
    saved_file = Path(viz.output_dir) / save_path
    assert saved_file.exists()


def test_plot_correlation_matrix(viz, sample_df):
    """Test correlation matrix plotting."""
    numeric_df = sample_df.select_dtypes(include=[np.number])
    fig = viz.plot_correlation_matrix(numeric_df)

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_time_series(viz, sample_df):
    """Test time series plotting."""
    sample_df["date"] = pd.date_range("2024-01-01", periods=len(sample_df))
    fig = viz.plot_time_series(
        sample_df, "date", "engagement_rate", "Engagement Over Time"
    )

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_time_series_with_hue(viz, sample_df):
    """Test time series plotting with grouping."""
    sample_df["date"] = pd.date_range("2024-01-01", periods=len(sample_df))
    fig = viz.plot_time_series(
        sample_df, "date", "engagement_rate", "Engagement by Platform", hue="platform"
    )

    assert fig is not None


def test_plot_hourly_heatmap(viz, sample_df):
    """Test hourly heatmap plotting."""
    fig = viz.plot_hourly_heatmap(sample_df, "engagement_rate")

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_platform_comparison_box(viz, sample_df):
    """Test platform comparison with box plot."""
    fig = viz.plot_platform_comparison(sample_df, "engagement_rate", plot_type="box")

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_platform_comparison_violin(viz, sample_df):
    """Test platform comparison with violin plot."""
    fig = viz.plot_platform_comparison(sample_df, "engagement_rate", plot_type="violin")

    assert fig is not None


def test_plot_scatter_with_regression(viz, sample_df):
    """Test scatter plot with regression line."""
    fig = viz.plot_scatter_with_regression(
        sample_df, "content_length", "engagement_rate", "Length vs Engagement"
    )

    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_bar_chart(viz):
    """Test bar chart plotting."""
    data = pd.Series([10, 20, 30], index=["A", "B", "C"])
    fig = viz.plot_bar_chart(data, "Test Bar Chart")

    assert fig is not None
    assert len(fig.axes) > 0