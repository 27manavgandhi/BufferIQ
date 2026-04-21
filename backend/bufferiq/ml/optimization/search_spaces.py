"""Predefined search spaces for hyperparameter optimization."""

from typing import Any, Dict, List

from scipy.stats import loguniform, randint, uniform

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

# Try to import skopt spaces
try:
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available, Bayesian spaces unavailable")


# ============================================================================
# XGBoost Search Spaces
# ============================================================================

XGBOOST_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0, 0.01, 0.1, 1],
}

XGBOOST_RANDOM = {
    "learning_rate": loguniform(0.01, 0.3),
    "max_depth": randint(3, 10),
    "n_estimators": randint(100, 500),
    "subsample": uniform(0.6, 0.4),  # uniform(low, width) -> [0.6, 1.0]
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
    "reg_alpha": loguniform(0.001, 10),
    "reg_lambda": loguniform(0.001, 10),
}

# Bayesian spaces (only available if skopt is installed)
if SKOPT_AVAILABLE:
    XGBOOST_BAYESIAN = {
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "max_depth": Integer(3, 10),
        "n_estimators": Integer(100, 500),
        "subsample": Real(0.6, 1.0, prior="uniform"),
        "colsample_bytree": Real(0.6, 1.0, prior="uniform"),
        "min_child_weight": Integer(1, 10),
        "gamma": Real(0, 0.5, prior="uniform"),
        "reg_alpha": Real(0.001, 10, prior="log-uniform"),
        "reg_lambda": Real(0.001, 10, prior="log-uniform"),
    }
else:
    XGBOOST_BAYESIAN = {}


# ============================================================================
# LightGBM Search Spaces
# ============================================================================

LIGHTGBM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],
    "n_estimators": [100, 200, 300],
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_freq": [0, 5, 10],
    "min_child_samples": [5, 10, 20],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0, 0.01, 0.1, 1],
}

LIGHTGBM_RANDOM = {
    "learning_rate": loguniform(0.01, 0.3),
    "num_leaves": randint(15, 150),
    "n_estimators": randint(100, 500),
    "feature_fraction": uniform(0.6, 0.4),
    "bagging_fraction": uniform(0.6, 0.4),
    "bagging_freq": randint(0, 10),
    "min_child_samples": randint(5, 30),
    "reg_alpha": loguniform(0.001, 10),
    "reg_lambda": loguniform(0.001, 10),
}

if SKOPT_AVAILABLE:
    LIGHTGBM_BAYESIAN = {
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "num_leaves": Integer(15, 150),
        "n_estimators": Integer(100, 500),
        "feature_fraction": Real(0.6, 1.0, prior="uniform"),
        "bagging_fraction": Real(0.6, 1.0, prior="uniform"),
        "bagging_freq": Integer(0, 10),
        "min_child_samples": Integer(5, 30),
        "reg_alpha": Real(0.001, 10, prior="log-uniform"),
        "reg_lambda": Real(0.001, 10, prior="log-uniform"),
    }
else:
    LIGHTGBM_BAYESIAN = {}


# ============================================================================
# RandomForest Search Spaces
# ============================================================================

RANDOMFOREST_GRID = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

RANDOMFOREST_RANDOM = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(5, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

if SKOPT_AVAILABLE:
    RANDOMFOREST_BAYESIAN = {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(5, 30),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }
else:
    RANDOMFOREST_BAYESIAN = {}


# ============================================================================
# Search Space Registry
# ============================================================================

class SearchSpaceRegistry:
    """Registry for model-specific search spaces."""

    # Grid search spaces
    _GRID_SPACES = {
        "xgboost": XGBOOST_GRID,
        "lightgbm": LIGHTGBM_GRID,
        "random_forest": RANDOMFOREST_GRID,
    }

    # Random search spaces
    _RANDOM_SPACES = {
        "xgboost": XGBOOST_RANDOM,
        "lightgbm": LIGHTGBM_RANDOM,
        "random_forest": RANDOMFOREST_RANDOM,
    }

    # Bayesian search spaces
    _BAYESIAN_SPACES = {
        "xgboost": XGBOOST_BAYESIAN,
        "lightgbm": LIGHTGBM_BAYESIAN,
        "random_forest": RANDOMFOREST_BAYESIAN,
    }

    @classmethod
    def get_search_space(
        cls, model_type: str, strategy: str
    ) -> Dict[str, Any]:
        """
        Get search space for a model type and strategy.
        
        Args:
            model_type: Model type ('xgboost', 'lightgbm', 'random_forest')
            strategy: Search strategy ('grid', 'random', 'bayesian')
        
        Returns:
            Search space dictionary
        
        Raises:
            ValueError: If model_type or strategy is invalid
        
        Example:
            >>> space = SearchSpaceRegistry.get_search_space('xgboost', 'grid')
            >>> print(space['learning_rate'])
            [0.01, 0.05, 0.1, 0.2]
        """
        if model_type not in cls._GRID_SPACES:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Supported: {list(cls._GRID_SPACES.keys())}"
            )

        if strategy == "grid":
            return cls._GRID_SPACES[model_type].copy()
        elif strategy == "random":
            return cls._RANDOM_SPACES[model_type].copy()
        elif strategy == "bayesian":
            if not SKOPT_AVAILABLE:
                raise ImportError(
                    "scikit-optimize required for Bayesian search. "
                    "Install: pip install scikit-optimize"
                )
            return cls._BAYESIAN_SPACES[model_type].copy()
        else:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Supported: ['grid', 'random', 'bayesian']"
            )

    @classmethod
    def list_model_types(cls) -> List[str]:
        """Get list of supported model types."""
        return list(cls._GRID_SPACES.keys())

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of supported search strategies."""
        strategies = ["grid", "random"]
        if SKOPT_AVAILABLE:
            strategies.append("bayesian")
        return strategies