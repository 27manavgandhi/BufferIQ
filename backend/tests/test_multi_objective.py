"""Tests for multi-objective optimization."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.multi_objective import MultiObjectiveOptimizer


@pytest.fixture
def model():
    """Create test model."""
    return RandomForestRegressor(n_estimators=10, random_state=42)


@pytest.fixture
def search_space():
    """Create test search space."""
    return {
        "max_depth": {"type": "int", "low": 5, "high": 15},
        "n_estimators": {"type": "int", "low": 10, "high": 50},
    }


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_multi_objective_initialization(model, search_space):
    """Test multi-objective optimizer initialization."""
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
        n_trials=10,
    )
    assert optimizer.metrics == ["r2", "training_time"]
    assert optimizer.directions == ["maximize", "minimize"]
    assert optimizer.n_trials == 10


def test_mismatched_metrics_directions(model, search_space):
    """Test mismatched metrics and directions raises error."""
    with pytest.raises(ValueError, match="same length"):
        MultiObjectiveOptimizer(
            model=model,
            search_space=search_space,
            metrics=["r2", "training_time"],
            directions=["maximize"],  # Mismatch
            n_trials=10,
        )


@patch("optuna.create_study")
def test_multi_objective_search(mock_create_study, model, search_space, sample_data):
    """Test multi-objective search."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_trials = [
        MagicMock(values=[0.75, 10.0]),
        MagicMock(values=[0.78, 15.0]),
        MagicMock(values=[0.72, 8.0]),
    ]
    mock_create_study.return_value = mock_study
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
        n_trials=10,
    )
    
    results = optimizer.search(X, y)
    
    assert "pareto_trials" in results
    assert "n_pareto_solutions" in results
    assert results["n_pareto_solutions"] == 3


@patch("optuna.create_study")
def test_multi_objective_with_three_metrics(
    mock_create_study, model, search_space, sample_data
):
    """Test multi-objective with three metrics."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_trials = [
        MagicMock(values=[0.75, 10.0, 2.0]),
    ]
    mock_create_study.return_value = mock_study
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time", "model_size"],
        directions=["maximize", "minimize", "minimize"],
        n_trials=5,
    )
    
    results = optimizer.search(X, y)
    
    assert results["n_pareto_solutions"] == 1


def test_suggest_params(model, search_space):
    """Test parameter suggestion."""
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
    )
    
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 10
    
    params = optimizer._suggest_params(mock_trial)
    
    assert "max_depth" in params
    assert "n_estimators" in params


def test_objective_returns_tuple(model, search_space, sample_data):
    """Test objective returns tuple of metric values."""
    X, y = sample_data
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
    )
    optimizer.X = X
    optimizer.y = y
    
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 10
    
    result = optimizer._objective(mock_trial)
    
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_unknown_metric_raises_error(model, search_space, sample_data):
    """Test unknown metric raises error."""
    X, y = sample_data
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["unknown_metric"],
        directions=["maximize"],
    )
    optimizer.X = X
    optimizer.y = y
    
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 10
    
    with pytest.raises(ValueError, match="Unknown metric"):
        optimizer._objective(mock_trial)


@patch("optuna.create_study")
@patch("plotly.graph_objects.Figure.write_html")
def test_visualize_pareto_front(
    mock_write_html, mock_create_study, model, search_space, sample_data, tmp_path
):
    """Test Pareto front visualization."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_trials = [
        MagicMock(values=[0.75, 10.0]),
        MagicMock(values=[0.78, 15.0]),
    ]
    mock_create_study.return_value = mock_study
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
        n_trials=5,
    )
    
    optimizer.search(X, y)
    
    save_path = tmp_path / "pareto.html"
    optimizer.visualize_pareto_front(save_path)
    
    assert mock_write_html.called


def test_visualize_without_search_raises_error(model, search_space, tmp_path):
    """Test visualize without search raises error."""
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
    )
    
    save_path = tmp_path / "pareto.html"
    
    with pytest.raises(ValueError, match="No study available"):
        optimizer.visualize_pareto_front(save_path)


@patch("optuna.create_study")
def test_visualize_with_empty_pareto(
    mock_create_study, model, search_space, sample_data, tmp_path
):
    """Test visualize with empty Pareto front."""
    X, y = sample_data
    
    # Setup mock with no Pareto solutions
    mock_study = MagicMock()
    mock_study.best_trials = []
    mock_create_study.return_value = mock_study
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
        n_trials=5,
    )
    
    optimizer.search(X, y)
    
    save_path = tmp_path / "pareto.html"
    optimizer.visualize_pareto_front(save_path)  # Should not raise


@patch("optuna.create_study")
def test_multi_objective_uses_nsga2(mock_create_study, model, search_space, sample_data):
    """Test multi-objective uses NSGA-II sampler."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_trials = []
    mock_create_study.return_value = mock_study
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=search_space,
        metrics=["r2", "training_time"],
        directions=["maximize", "minimize"],
        n_trials=5,
    )
    
    optimizer.search(X, y)
    
    # Verify NSGA-II sampler was used
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["sampler"] is not None