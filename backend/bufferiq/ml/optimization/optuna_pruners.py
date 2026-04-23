"""Optuna pruner registry and factory."""

from typing import Any, Dict, List

import optuna
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
)

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class PrunerRegistry:
    """
    Centralized registry for Optuna pruners.
    
    Provides factory methods to create pruners by name with
    appropriate default parameters.
    """

    @staticmethod
    def get_pruner(name: str, **kwargs: Any) -> BasePruner:
        """
        Get pruner by name.
        
        Args:
            name: Pruner name ('median', 'hyperband', 'percentile',
                  'successive_halving', 'nop')
            **kwargs: Pruner-specific parameters
        
        Returns:
            Configured pruner instance
        
        Raises:
            ValueError: If pruner name is unknown
        
        Example:
            >>> pruner = PrunerRegistry.get_pruner('median', n_startup_trials=5)
            >>> isinstance(pruner, MedianPruner)
            True
        """
        if name == "median":
            return MedianPruner(
                n_startup_trials=kwargs.get("n_startup_trials", 5),
                n_warmup_steps=kwargs.get("n_warmup_steps", 0),
                interval_steps=kwargs.get("interval_steps", 1),
            )
        elif name == "hyperband":
            return HyperbandPruner(
                min_resource=kwargs.get("min_resource", 1),
                max_resource=kwargs.get("max_resource", "auto"),
                reduction_factor=kwargs.get("reduction_factor", 3),
            )
        elif name == "percentile":
            return PercentilePruner(
                percentile=kwargs.get("percentile", 25.0),
                n_startup_trials=kwargs.get("n_startup_trials", 5),
                n_warmup_steps=kwargs.get("n_warmup_steps", 0),
                interval_steps=kwargs.get("interval_steps", 1),
            )
        elif name == "successive_halving":
            return SuccessiveHalvingPruner(
                min_resource=kwargs.get("min_resource", 1),
                reduction_factor=kwargs.get("reduction_factor", 4),
                min_early_stopping_rate=kwargs.get("min_early_stopping_rate", 0),
            )
        elif name == "nop":
            return NopPruner()
        else:
            raise ValueError(
                f"Unknown pruner: {name}. "
                f"Supported: {PrunerRegistry.list_pruners()}"
            )

    @staticmethod
    def list_pruners() -> List[str]:
        """
        Get list of available pruner names.
        
        Returns:
            List of pruner names
        """
        return ["median", "hyperband", "percentile", "successive_halving", "nop"]

    @staticmethod
    def get_default_config(name: str) -> Dict[str, Any]:
        """
        Get default configuration for a pruner.
        
        Args:
            name: Pruner name
        
        Returns:
            Dictionary of default parameters
        """
        configs = {
            "median": {
                "n_startup_trials": 5,
                "n_warmup_steps": 0,
                "interval_steps": 1,
            },
            "hyperband": {
                "min_resource": 1,
                "max_resource": "auto",
                "reduction_factor": 3,
            },
            "percentile": {
                "percentile": 25.0,
                "n_startup_trials": 5,
                "n_warmup_steps": 0,
                "interval_steps": 1,
            },
            "successive_halving": {
                "min_resource": 1,
                "reduction_factor": 4,
                "min_early_stopping_rate": 0,
            },
            "nop": {},
        }
        
        if name not in configs:
            raise ValueError(f"Unknown pruner: {name}")
        
        return configs[name]