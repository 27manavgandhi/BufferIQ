"""Tests for Optuna pruner registry."""

import pytest
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
)

from bufferiq.ml.optimization.optuna_pruners import PrunerRegistry


def test_get_median_pruner():
    """Test getting median pruner."""
    pruner = PrunerRegistry.get_pruner("median")
    assert isinstance(pruner, MedianPruner)


def test_get_median_pruner_with_params():
    """Test median pruner with custom parameters."""
    pruner = PrunerRegistry.get_pruner("median", n_startup_trials=10)
    assert isinstance(pruner, MedianPruner)


def test_get_hyperband_pruner():
    """Test getting Hyperband pruner."""
    pruner = PrunerRegistry.get_pruner("hyperband")
    assert isinstance(pruner, HyperbandPruner)


def test_get_hyperband_pruner_with_params():
    """Test Hyperband pruner with custom parameters."""
    pruner = PrunerRegistry.get_pruner("hyperband", reduction_factor=4)
    assert isinstance(pruner, HyperbandPruner)


def test_get_percentile_pruner():
    """Test getting percentile pruner."""
    pruner = PrunerRegistry.get_pruner("percentile")
    assert isinstance(pruner, PercentilePruner)


def test_get_percentile_pruner_with_params():
    """Test percentile pruner with custom parameters."""
    pruner = PrunerRegistry.get_pruner("percentile", percentile=30.0)
    assert isinstance(pruner, PercentilePruner)


def test_get_successive_halving_pruner():
    """Test getting successive halving pruner."""
    pruner = PrunerRegistry.get_pruner("successive_halving")
    assert isinstance(pruner, SuccessiveHalvingPruner)


def test_get_nop_pruner():
    """Test getting no-op pruner."""
    pruner = PrunerRegistry.get_pruner("nop")
    assert isinstance(pruner, NopPruner)


def test_get_unknown_pruner():
    """Test unknown pruner raises error."""
    with pytest.raises(ValueError, match="Unknown pruner"):
        PrunerRegistry.get_pruner("invalid_pruner")


def test_list_pruners():
    """Test listing available pruners."""
    pruners = PrunerRegistry.list_pruners()
    assert "median" in pruners
    assert "hyperband" in pruners
    assert "percentile" in pruners
    assert "successive_halving" in pruners
    assert "nop" in pruners
    assert len(pruners) == 5


def test_get_default_config_median():
    """Test default config for median pruner."""
    config = PrunerRegistry.get_default_config("median")
    assert "n_startup_trials" in config
    assert "n_warmup_steps" in config
    assert "interval_steps" in config


def test_get_default_config_hyperband():
    """Test default config for Hyperband pruner."""
    config = PrunerRegistry.get_default_config("hyperband")
    assert "min_resource" in config
    assert "max_resource" in config
    assert "reduction_factor" in config


def test_get_default_config_unknown():
    """Test unknown pruner config raises error."""
    with pytest.raises(ValueError, match="Unknown pruner"):
        PrunerRegistry.get_default_config("invalid_pruner")