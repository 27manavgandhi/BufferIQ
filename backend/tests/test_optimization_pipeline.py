"""Tests for optimization pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bufferiq.ml.optimization.config_schema import OptimizationConfig
from bufferiq.ml.optimization.pipeline import OptimizationPipeline


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        model_type="xgboost",
        strategy="grid",
        cv_folds=3,
        output_dir=Path("outputs/test_optimization"),
    )


@pytest.fixture
def pipeline(config):
    """Create optimization pipeline."""
    return OptimizationPipeline(config)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_pipeline_initialization(config):
    """Test pipeline initialization."""
    pipeline = OptimizationPipeline(config)
    assert pipeline.config == config
    assert pipeline.tracker is not None


def test_pipeline_get_trainer_xgboost(config):
    """Test pipeline creates XGBoost trainer."""
    pipeline = OptimizationPipeline(config)
    trainer = pipeline._get_trainer()
    
    from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer
    assert isinstance(trainer, XGBoostTrainer)


def test_pipeline_get_trainer_lightgbm():
    """Test pipeline creates LightGBM trainer."""
    config = OptimizationConfig(
        model_type="lightgbm",
        strategy="grid",
    )
    pipeline = OptimizationPipeline(config)
    trainer = pipeline._get_trainer()
    
    from bufferiq.ml.trainers.lightgbm_trainer import LightGBMTrainer
    assert isinstance(trainer, LightGBMTrainer)


def test_pipeline_get_trainer_random_forest():
    """Test pipeline creates RandomForest trainer."""
    config = OptimizationConfig(
        model_type="random_forest",
        strategy="grid",
    )
    pipeline = OptimizationPipeline(config)
    trainer = pipeline._get_trainer()
    
    from bufferiq.ml.trainers.random_forest_trainer import RandomForestTrainer
    assert isinstance(trainer, RandomForestTrainer)


def test_pipeline_get_search_space_default(pipeline):
    """Test pipeline gets default search space."""
    space = pipeline._get_search_space()
    assert space is not None
    assert "learning_rate" in space


def test_pipeline_get_search_space_custom():
    """Test pipeline uses custom search space."""
    custom_space = {"learning_rate": [0.1, 0.2], "max_depth": [3, 5]}
    config = OptimizationConfig(
        model_type="xgboost",
        strategy="grid",
        search_space=custom_space,
    )
    pipeline = OptimizationPipeline(config)
    
    space = pipeline._get_search_space()
    assert space == custom_space


def test_pipeline_get_optimizer_grid(pipeline):
    """Test pipeline creates grid search optimizer."""
    search_space = {"learning_rate": [0.1, 0.2]}
    optimizer = pipeline._get_optimizer(search_space)
    
    from bufferiq.ml.optimization.grid_search import GridSearchOptimizer
    assert isinstance(optimizer, GridSearchOptimizer)


def test_pipeline_get_optimizer_random():
    """Test pipeline creates random search optimizer."""
    from scipy.stats import uniform
    
    config = OptimizationConfig(
        model_type="xgboost",
        strategy="random",
        n_iter=10,
    )
    pipeline = OptimizationPipeline(config)
    search_space = {"learning_rate": uniform(0.01, 0.3)}
    optimizer = pipeline._get_optimizer(search_space)
    
    from bufferiq.ml.optimization.random_search import RandomSearchOptimizer
    assert isinstance(optimizer, RandomSearchOptimizer)


@patch("bufferiq.ml.optimization.grid_search.GridSearchOptimizer.search")
@pytest.mark.asyncio
async def test_pipeline_run_success(mock_search, pipeline, sample_data):
    """Test pipeline runs successfully."""
    X, y = sample_data
    
    # Setup mock
    mock_search.return_value = {
        "best_params": {"learning_rate": 0.1},
        "best_score": 0.75,
        "cv_results": {
            "params": [{"learning_rate": 0.1}],
            "mean_test_score": [0.75],
            "mean_fit_time": [10.0],
        },
        "total_trials": 1,
    }
    
    results = await pipeline.run(X, y)
    
    assert "best_params" in results
    assert "best_score" in results
    assert results["best_score"] == 0.75


@patch("bufferiq.ml.optimization.grid_search.GridSearchOptimizer.search")
@pytest.mark.asyncio
async def test_pipeline_run_logs_trials(mock_search, pipeline, sample_data):
    """Test pipeline logs all trials."""
    X, y = sample_data
    
    # Setup mock with multiple trials
    mock_search.return_value = {
        "best_params": {"learning_rate": 0.2},
        "best_score": 0.78,
        "cv_results": {
            "params": [
                {"learning_rate": 0.1},
                {"learning_rate": 0.2},
            ],
            "mean_test_score": [0.75, 0.78],
            "mean_fit_time": [10.0, 12.0],
        },
        "total_trials": 2,
    }
    
    await pipeline.run(X, y)
    
    # Check trials were logged
    assert len(pipeline.tracker.trials) == 2


@patch("bufferiq.ml.optimization.grid_search.GridSearchOptimizer.search")
@pytest.mark.asyncio
async def test_pipeline_run_saves_results(mock_search, pipeline, sample_data, tmp_path):
    """Test pipeline saves results to files."""
    X, y = sample_data
    pipeline.config.output_dir = tmp_path
    pipeline.tracker.output_dir = tmp_path
    
    # Setup mock
    mock_search.return_value = {
        "best_params": {"learning_rate": 0.1},
        "best_score": 0.75,
        "cv_results": {
            "params": [{"learning_rate": 0.1}],
            "mean_test_score": [0.75],
            "mean_fit_time": [10.0],
        },
        "total_trials": 1,
    }
    
    results = await pipeline.run(X, y)
    
    # Check files were created
    assert (tmp_path / "trials.json").exists()
    assert (tmp_path / "best_params.yaml").exists()
    assert (tmp_path / "optimization_report.json").exists()


@pytest.mark.asyncio
async def test_pipeline_run_uses_dummy_data_when_none_provided(pipeline):
    """Test pipeline uses dummy data when none provided."""
    results = await pipeline.run()
    
    # Should complete without error
    assert "best_params" in results


@patch("bufferiq.ml.optimization.grid_search.GridSearchOptimizer.search")
@pytest.mark.asyncio
async def test_pipeline_run_handles_errors(mock_search, pipeline, sample_data):
    """Test pipeline handles errors gracefully."""
    X, y = sample_data
    
    # Setup mock to raise error
    mock_search.side_effect = RuntimeError("Test error")
    
    with pytest.raises(RuntimeError):
        await pipeline.run(X, y)