"""Tests for parallel Optuna optimizer."""

from unittest.mock import MagicMock, patch

import pytest

from bufferiq.ml.optimization.parallel_optimizer import ParallelOptimizer


def test_parallel_optimizer_initialization():
    """Test parallel optimizer initialization."""
    def objective(trial):
        return trial.suggest_float("x", 0, 1)
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=4,
        n_trials_per_worker=25,
    )
    
    assert optimizer.n_workers == 4
    assert optimizer.n_trials_per_worker == 25


@patch("optuna.create_study")
@patch("optuna.load_study")
@patch("multiprocessing.Pool")
def test_parallel_run(mock_pool, mock_load_study, mock_create_study):
    """Test parallel optimization run."""
    def objective(trial):
        return trial.suggest_float("x", 0, 1)
    
    # Setup mocks
    mock_study = MagicMock()
    mock_study.trials = [MagicMock() for _ in range(100)]
    mock_create_study.return_value = mock_study
    mock_load_study.return_value = mock_study
    
    mock_pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=2,
        n_trials_per_worker=10,
    )
    
    study = optimizer.run()
    
    assert mock_create_study.called
    assert mock_pool_instance.map.called
    assert study == mock_study


@patch("optuna.create_study")
@patch("optuna.load_study")
@patch("multiprocessing.Pool")
def test_parallel_run_with_sampler(mock_pool, mock_load_study, mock_create_study):
    """Test parallel run with custom sampler."""
    def objective(trial):
        return trial.suggest_float("x", 0, 1)
    
    # Setup mocks
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    mock_load_study.return_value = mock_study
    
    mock_pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    
    from optuna.samplers import RandomSampler
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=2,
    )
    
    sampler = RandomSampler(seed=42)
    optimizer.run(sampler=sampler)
    
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["sampler"] == sampler


@patch("optuna.load_study")
def test_worker_function(mock_load_study):
    """Test worker function."""
    def objective(trial):
        return trial.suggest_float("x", 0, 1)
    
    # Setup mock
    mock_study = MagicMock()
    mock_load_study.return_value = mock_study
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=1,
        n_trials_per_worker=5,
    )
    
    # Run worker (should not raise)
    optimizer._worker(0)
    
    assert mock_load_study.called
    assert mock_study.optimize.called


@patch("optuna.load_study")
def test_worker_handles_exception(mock_load_study):
    """Test worker handles exceptions."""
    def objective(trial):
        raise RuntimeError("Test error")
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.optimize.side_effect = RuntimeError("Test error")
    mock_load_study.return_value = mock_study
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=1,
        n_trials_per_worker=5,
    )
    
    # Worker should handle exception gracefully
    optimizer._worker(0)


@patch("optuna.create_study")
@patch("optuna.load_study")
@patch("multiprocessing.Pool")
def test_parallel_run_with_pruner(mock_pool, mock_load_study, mock_create_study):
    """Test parallel run with pruner."""
    def objective(trial):
        return trial.suggest_float("x", 0, 1)
    
    # Setup mocks
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    mock_load_study.return_value = mock_study
    
    mock_pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    
    from optuna.pruners import MedianPruner
    
    optimizer = ParallelOptimizer(
        objective=objective,
        study_name="test_study",
        storage="sqlite:///test.db",
        n_workers=2,
    )
    
    pruner = MedianPruner()
    optimizer.run(pruner=pruner)
    
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["pruner"] == pruner