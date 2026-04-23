"""Parallel Optuna optimization using multiprocessing."""

import multiprocessing as mp
from typing import Callable, Optional

import optuna

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ParallelOptimizer:
    """
    Run Optuna optimization in parallel.
    
    Executes multiple workers simultaneously, each running trials
    on a shared study stored in persistent storage.
    """

    def __init__(
        self,
        objective: Callable,
        study_name: str,
        storage: str,
        n_workers: int = 4,
        n_trials_per_worker: int = 25,
    ) -> None:
        """
        Initialize parallel optimizer.
        
        Args:
            objective: Objective function to optimize
            study_name: Name of shared study
            storage: Storage URL (must be thread-safe, e.g., SQLite or PostgreSQL)
            n_workers: Number of parallel workers
            n_trials_per_worker: Number of trials each worker should run
        
        Example:
            >>> def objective(trial):
            ...     x = trial.suggest_float('x', -10, 10)
            ...     return x ** 2
            >>> parallel_opt = ParallelOptimizer(
            ...     objective=objective,
            ...     study_name="parallel_study",
            ...     storage="sqlite:///parallel.db",
            ...     n_workers=4
            ... )
            >>> study = parallel_opt.run()
        """
        self.objective = objective
        self.study_name = study_name
        self.storage = storage
        self.n_workers = n_workers
        self.n_trials_per_worker = n_trials_per_worker
        
        logger.info(
            f"Parallel optimizer initialized: {n_workers} workers, "
            f"{n_trials_per_worker} trials per worker"
        )

    def _worker(self, worker_id: int) -> None:
        """
        Worker function for parallel optimization.
        
        Args:
            worker_id: Worker identifier
        """
        try:
            # Load shared study
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
            )
            
            logger.info(f"Worker {worker_id} started")
            
            # Run trials
            study.optimize(
                self.objective,
                n_trials=self.n_trials_per_worker,
                show_progress_bar=False,
            )
            
            logger.info(
                f"Worker {worker_id} completed {self.n_trials_per_worker} trials"
            )
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)

    def run(
        self,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """
        Run parallel optimization.
        
        Args:
            direction: Optimization direction
            sampler: Optuna sampler
            pruner: Optuna pruner
        
        Returns:
            Completed study object
        
        Example:
            >>> study = parallel_opt.run(direction="minimize")
            >>> print(f"Best value: {study.best_value}")
        """
        # Create study
        logger.info(f"Creating study: {self.study_name}")
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        # Run workers in parallel
        logger.info(f"Starting {self.n_workers} parallel workers")
        
        with mp.Pool(processes=self.n_workers) as pool:
            pool.map(self._worker, range(self.n_workers))
        
        logger.info("All workers completed")
        
        # Load final study
        final_study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
        )
        
        logger.info(
            f"Parallel optimization complete: {len(final_study.trials)} total trials"
        )
        
        return final_study