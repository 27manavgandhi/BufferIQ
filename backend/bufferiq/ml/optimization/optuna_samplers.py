"""Optuna sampler registry and factory."""

from typing import Any

from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    GridSampler,
    NSGAIISampler,
    RandomSampler,
    TPESampler,
)

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class SamplerRegistry:
    """
    Centralized registry for Optuna samplers.

    Provides factory methods to create samplers by name with
    appropriate default parameters.
    """

    @staticmethod
    def get_sampler(name: str, seed: int = 42, **kwargs: Any) -> BaseSampler:
        """
        Get sampler by name.

        Args:
            name: Sampler name ('tpe', 'random', 'grid', 'cmaes', 'nsga2')
            seed: Random seed for reproducibility
            **kwargs: Sampler-specific parameters

        Returns:
            Configured sampler instance

        Raises:
            ValueError: If sampler name is unknown

        Example:
            >>> sampler = SamplerRegistry.get_sampler('tpe', seed=42)
            >>> isinstance(sampler, TPESampler)
            True
        """
        if name == "tpe":
            return TPESampler(
                seed=seed,
                n_startup_trials=kwargs.get("n_startup_trials", 10),
                n_ei_candidates=kwargs.get("n_ei_candidates", 24),
                multivariate=kwargs.get("multivariate", False),
            )
        elif name == "random":
            return RandomSampler(seed=seed)
        elif name == "grid":
            search_space = kwargs.get("search_space")
            if search_space is None:
                raise ValueError("Grid sampler requires 'search_space' parameter")
            return GridSampler(search_space, seed=seed)
        elif name == "cmaes":
            return CmaEsSampler(
                seed=seed,
                n_startup_trials=kwargs.get("n_startup_trials", 1),
            )
        elif name == "nsga2":
            return NSGAIISampler(
                seed=seed,
                population_size=kwargs.get("population_size", 50),
                mutation_prob=kwargs.get("mutation_prob", None),
                crossover_prob=kwargs.get("crossover_prob", 0.9),
            )
        else:
            raise ValueError(
                f"Unknown sampler: {name}. "
                f"Supported: {SamplerRegistry.list_samplers()}"
            )

    @staticmethod
    def list_samplers() -> list[str]:
        """
        Get list of available sampler names.

        Returns:
            List of sampler names
        """
        return ["tpe", "random", "grid", "cmaes", "nsga2"]

    @staticmethod
    def get_default_config(name: str) -> dict[str, Any]:
        """
        Get default configuration for a sampler.

        Args:
            name: Sampler name

        Returns:
            Dictionary of default parameters
        """
        configs = {
            "tpe": {
                "n_startup_trials": 10,
                "n_ei_candidates": 24,
                "multivariate": False,
            },
            "random": {},
            "grid": {},
            "cmaes": {
                "n_startup_trials": 1,
            },
            "nsga2": {
                "population_size": 50,
                "mutation_prob": None,
                "crossover_prob": 0.9,
            },
        }

        if name not in configs:
            raise ValueError(f"Unknown sampler: {name}")

        return configs[name]
