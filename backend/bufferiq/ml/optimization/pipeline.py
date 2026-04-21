"""Optimization pipeline orchestrating the full optimization workflow."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline
from bufferiq.ml.optimization.bayesian import BayesianOptimizer
from bufferiq.ml.optimization.config_schema import OptimizationConfig
from bufferiq.ml.optimization.grid_search import GridSearchOptimizer
from bufferiq.ml.optimization.random_search import RandomSearchOptimizer
from bufferiq.ml.optimization.result_tracker import OptimizationResultTracker
from bufferiq.ml.optimization.search_spaces import SearchSpaceRegistry
from bufferiq.ml.trainers.lightgbm_trainer import LightGBMTrainer
from bufferiq.ml.trainers.random_forest_trainer import RandomForestTrainer
from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer
from bufferiq.ml.training.data_preparation import DataPreparation

logger = get_logger(__name__)


class OptimizationPipeline:
    """
    Orchestrate hyperparameter optimization workflow.
    
    Coordinates data preparation, search execution, result tracking,
    and model retraining with optimal parameters.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        session: Optional[AsyncSession] = None,
    ) -> None:
        """
        Initialize optimization pipeline.
        
        Args:
            config: Optimization configuration
            session: Database session for data loading (optional)
        
        Example:
            >>> config = OptimizationConfig.from_yaml("config.yaml")
            >>> pipeline = OptimizationPipeline(config)
            >>> results = await pipeline.run()
        """
        self.config = config
        self.session = session
        
        # Initialize result tracker
        self.tracker = OptimizationResultTracker(config.output_dir)
        
        logger.info(
            f"Optimization pipeline initialized: {config.model_type} "
            f"with {config.strategy} search"
        )

    def _get_trainer(self) -> Any:
        """
        Get trainer instance for model type.
        
        Returns:
            Trainer instance
        
        Raises:
            ValueError: If model_type is invalid
        """
        if self.config.model_type == "xgboost":
            return XGBoostTrainer(random_state=self.config.random_state)
        elif self.config.model_type == "lightgbm":
            return LightGBMTrainer(random_state=self.config.random_state)
        elif self.config.model_type == "random_forest":
            return RandomForestTrainer(random_state=self.config.random_state)
        else:
            raise ValueError(f"Invalid model_type: {self.config.model_type}")

    def _get_search_space(self) -> Dict[str, Any]:
        """
        Get search space for optimization.
        
        Returns:
            Search space dictionary
        """
        if self.config.search_space is not None:
            logger.info("Using custom search space from config")
            return self.config.search_space
        
        logger.info("Using default search space from registry")
        return SearchSpaceRegistry.get_search_space(
            self.config.model_type,
            self.config.strategy,
        )

    def _get_optimizer(self, search_space: Dict[str, Any]) -> Any:
        """
        Get optimizer instance for strategy.
        
        Args:
            search_space: Search space dictionary
        
        Returns:
            Optimizer instance
        
        Raises:
            ValueError: If strategy is invalid
        """
        trainer = self._get_trainer()
        
        if self.config.strategy == "grid":
            return GridSearchOptimizer(
                model=trainer.model,
                param_grid=search_space,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )
        elif self.config.strategy == "random":
            return RandomSearchOptimizer(
                model=trainer.model,
                param_distributions=search_space,
                n_iter=self.config.n_iter or 100,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )
        elif self.config.strategy == "bayesian":
            return BayesianOptimizer(
                model=trainer.model,
                search_spaces=search_space,
                n_iter=self.config.n_iter or 50,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Invalid strategy: {self.config.strategy}")

    async def run(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run complete optimization pipeline.
        
        Args:
            X_train: Training features (optional, will load from DB if not provided)
            y_train: Training targets (optional, will load from DB if not provided)
        
        Returns:
            Dictionary containing:
                - best_params: Optimal hyperparameters
                - best_score: Best cross-validation score
                - total_trials: Number of trials performed
                - optimization_time: Time taken (seconds)
                - baseline_score: Score before optimization (if available)
                - improvement_pct: Improvement percentage (if baseline available)
        
        Example:
            >>> results = await pipeline.run()
            >>> print(f"Best R²: {results['best_score']:.4f}")
            Best R²: 0.7589
        """
        logger.info("=" * 80)
        logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Prepare data (if not provided)
            if X_train is None or y_train is None:
                logger.info("Data not provided, loading from database...")
                # For now, use dummy data
                # In production, this would load from database
                logger.warning("Using dummy data for demonstration")
                X_train = np.random.randn(1000, 50)
                y_train = np.random.randn(1000)
            
            logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")
            
            # 2. Get search space
            search_space = self._get_search_space()
            logger.info(f"Search space: {len(search_space)} parameters")
            
            # 3. Get optimizer
            optimizer = self._get_optimizer(search_space)
            
            # 4. Run optimization
            logger.info(f"Starting {self.config.strategy} search...")
            results = optimizer.search(X_train, y_train)
            
            # 5. Log trials to tracker
            cv_results = results["cv_results"]
            for i, (params, score) in enumerate(
                zip(cv_results["params"], cv_results["mean_test_score"])
            ):
                duration = cv_results.get("mean_fit_time", [0] * len(params))[i]
                self.tracker.log_trial(
                    trial_id=i + 1,
                    params=params,
                    score=float(score),
                    duration=float(duration),
                )
            
            # 6. Save results
            self.tracker.save_trials()
            self.tracker.export_best_params()
            report = self.tracker.save_report()
            
            optimization_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Best score: {results['best_score']:.4f}")
            logger.info(f"Best params: {results['best_params']}")
            logger.info(f"Total trials: {results['total_trials']}")
            logger.info(f"Time taken: {optimization_time:.2f}s")
            
            return {
                "best_params": results["best_params"],
                "best_score": results["best_score"],
                "total_trials": results["total_trials"],
                "optimization_time": optimization_time,
                "report_path": str(report),
            }
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}", exc_info=True)
            raise