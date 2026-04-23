"""Tests for advanced optimization visualizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bufferiq.ml.optimization.advanced_visualizer import (
    AdvancedOptimizationVisualizer,
)


@pytest.fixture
def mock_study():
    """Create mock Optuna study."""
    study = MagicMock()
    study.study_name = "test_study"
    study.trials = [MagicMock() for _ in range(10)]
    return study


def test_visualizer_initialization(mock_study):
    """Test visualizer initialization."""
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    assert visualizer.study == mock_study


@patch("optuna.visualization.plot_optimization_history")
def test_plot_optimization_history(mock_plot, mock_study, tmp_path):
    """Test optimization history plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "history.html"
    visualizer.plot_optimization_history(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_param_importances")
def test_plot_param_importances(mock_plot, mock_study, tmp_path):
    """Test parameter importances plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "importances.html"
    visualizer.plot_param_importances(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_param_importances")
def test_plot_param_importances_handles_exception(mock_plot, mock_study, tmp_path):
    """Test param importances handles exceptions."""
    mock_plot.side_effect = RuntimeError("Test error")
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "importances.html"
    
    # Should not raise, just log warning
    visualizer.plot_param_importances(save_path)


@patch("optuna.visualization.plot_parallel_coordinate")
def test_plot_parallel_coordinate(mock_plot, mock_study, tmp_path):
    """Test parallel coordinate plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "parallel.html"
    visualizer.plot_parallel_coordinate(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_contour")
def test_plot_contour(mock_plot, mock_study, tmp_path):
    """Test contour plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "contour.html"
    visualizer.plot_contour(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_contour")
def test_plot_contour_handles_exception(mock_plot, mock_study, tmp_path):
    """Test contour plot handles exceptions."""
    mock_plot.side_effect = RuntimeError("Test error")
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "contour.html"
    
    # Should not raise
    visualizer.plot_contour(save_path)


@patch("optuna.visualization.plot_slice")
def test_plot_slice(mock_plot, mock_study, tmp_path):
    """Test slice plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "slice.html"
    visualizer.plot_slice(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_edf")
def test_plot_edf(mock_plot, mock_study, tmp_path):
    """Test EDF plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "edf.html"
    visualizer.plot_edf(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_timeline")
def test_plot_timeline(mock_plot, mock_study, tmp_path):
    """Test timeline plot."""
    mock_fig = MagicMock()
    mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    save_path = tmp_path / "timeline.html"
    visualizer.plot_timeline(save_path)
    
    assert mock_plot.called
    assert mock_fig.write_html.called


@patch("optuna.visualization.plot_optimization_history")
@patch("optuna.visualization.plot_param_importances")
@patch("optuna.visualization.plot_parallel_coordinate")
@patch("optuna.visualization.plot_contour")
@patch("optuna.visualization.plot_slice")
@patch("optuna.visualization.plot_edf")
@patch("optuna.visualization.plot_timeline")
def test_create_all_visualizations(
    mock_timeline,
    mock_edf,
    mock_slice,
    mock_contour,
    mock_parallel,
    mock_importance,
    mock_history,
    mock_study,
    tmp_path,
):
    """Test creating all visualizations."""
    # Setup all mocks
    for mock_plot in [
        mock_timeline,
        mock_edf,
        mock_slice,
        mock_contour,
        mock_parallel,
        mock_importance,
        mock_history,
    ]:
        mock_fig = MagicMock()
        mock_plot.return_value = mock_fig
    
    visualizer = AdvancedOptimizationVisualizer(mock_study)
    output_dir = tmp_path / "visualizations"
    visualizer.create_all_visualizations(output_dir)
    
    assert output_dir.exists()
    # All plots should be called
    assert mock_history.called
    assert mock_importance.called
    assert mock_parallel.called