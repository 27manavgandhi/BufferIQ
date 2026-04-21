"""Tests for search space registry."""

import pytest
from scipy.stats import randint, uniform

from bufferiq.ml.optimization.search_spaces import (
    LIGHTGBM_GRID,
    RANDOMFOREST_GRID,
    SKOPT_AVAILABLE,
    XGBOOST_GRID,
    SearchSpaceRegistry,
)


def test_xgboost_grid_exists():
    """Test XGBoost grid search space exists."""
    assert XGBOOST_GRID is not None
    assert "learning_rate" in XGBOOST_GRID
    assert "max_depth" in XGBOOST_GRID
    assert isinstance(XGBOOST_GRID["learning_rate"], list)


def test_lightgbm_grid_exists():
    """Test LightGBM grid search space exists."""
    assert LIGHTGBM_GRID is not None
    assert "learning_rate" in LIGHTGBM_GRID
    assert "num_leaves" in LIGHTGBM_GRID
    assert isinstance(LIGHTGBM_GRID["num_leaves"], list)


def test_randomforest_grid_exists():
    """Test RandomForest grid search space exists."""
    assert RANDOMFOREST_GRID is not None
    assert "n_estimators" in RANDOMFOREST_GRID
    assert "max_depth" in RANDOMFOREST_GRID
    assert isinstance(RANDOMFOREST_GRID["n_estimators"], list)


def test_registry_get_xgboost_grid():
    """Test registry returns XGBoost grid space."""
    space = SearchSpaceRegistry.get_search_space("xgboost", "grid")
    assert space is not None
    assert "learning_rate" in space
    assert isinstance(space["learning_rate"], list)


def test_registry_get_lightgbm_grid():
    """Test registry returns LightGBM grid space."""
    space = SearchSpaceRegistry.get_search_space("lightgbm", "grid")
    assert space is not None
    assert "num_leaves" in space
    assert isinstance(space["num_leaves"], list)


def test_registry_get_randomforest_grid():
    """Test registry returns RandomForest grid space."""
    space = SearchSpaceRegistry.get_search_space("random_forest", "grid")
    assert space is not None
    assert "n_estimators" in space
    assert isinstance(space["n_estimators"], list)


def test_registry_get_random_space():
    """Test registry returns random search space."""
    space = SearchSpaceRegistry.get_search_space("xgboost", "random")
    assert space is not None
    assert "learning_rate" in space


def test_registry_invalid_model_type():
    """Test registry rejects invalid model type."""
    with pytest.raises(ValueError, match="Invalid model_type"):
        SearchSpaceRegistry.get_search_space("invalid_model", "grid")


def test_registry_invalid_strategy():
    """Test registry rejects invalid strategy."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        SearchSpaceRegistry.get_search_space("xgboost", "invalid_strategy")


@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="skopt not installed")
def test_registry_get_bayesian_space():
    """Test registry returns Bayesian search space."""
    space = SearchSpaceRegistry.get_search_space("xgboost", "bayesian")
    assert space is not None
    assert "learning_rate" in space


@pytest.mark.skipif(SKOPT_AVAILABLE, reason="Test for when skopt not available")
def test_registry_bayesian_without_skopt():
    """Test registry rejects Bayesian without skopt."""
    with pytest.raises(ImportError, match="scikit-optimize required"):
        SearchSpaceRegistry.get_search_space("xgboost", "bayesian")


def test_registry_list_model_types():
    """Test registry lists model types."""
    model_types = SearchSpaceRegistry.list_model_types()
    assert "xgboost" in model_types
    assert "lightgbm" in model_types
    assert "random_forest" in model_types


def test_registry_list_strategies():
    """Test registry lists strategies."""
    strategies = SearchSpaceRegistry.list_strategies()
    assert "grid" in strategies
    assert "random" in strategies
    
    if SKOPT_AVAILABLE:
        assert "bayesian" in strategies


def test_registry_returns_copy():
    """Test registry returns copy, not reference."""
    space1 = SearchSpaceRegistry.get_search_space("xgboost", "grid")
    space2 = SearchSpaceRegistry.get_search_space("xgboost", "grid")
    
    # Modify space1
    space1["new_param"] = [1, 2, 3]
    
    # space2 should not be affected
    assert "new_param" not in space2