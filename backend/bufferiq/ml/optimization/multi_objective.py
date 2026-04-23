"""Multi-objective hyperparameter optimization using NSGA-II."""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class MultiObjectiveOptimizer:
    """
    Multi-objective hyperparameter optimization.
    
    Optimizes multiple metrics simultaneously (e.g., accuracy, speed, size)
    using NSGA-II algorithm to find Pareto-optimal solutions.
    """

    def __init__(
        self,
        model: BaseEstimator,
        search_space: Dict[str, Dict[str, Any]],
        metrics: List[str],
        directions: List[str],
        n_trials: int = 100,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Initialize multi-objective optimizer.
        
        Args:
            model: Scikit-learn compatible model
            search_space: Parameter search space
            metrics: List of metrics to optimize (e.g., ['r2', 'training_time'])
            directions: Optimization directions for each metric
                       ('maximize' or 'minimize')
            n_trials: Number of trials to run
            cv: Number of cross-validation folds
            random_state: Random seed
        
        Raises:
            ValueError: If metrics and directions have different lengths
        
        Example:
            >>> optimizer = MultiObjectiveOptimizer(
            ...     model=xgboost_model,
            ...     search_space=search_space,
            ...     metrics=['r2', 'training_time', 'model_size'],
            ...     directions=['maximize', 'minimize', 'minimize'],
            ...     n_trials=100
            ... )
            >>> results = optimizer.search(X_train, y_train)
        """
        if len(metrics) != len(directions):
            raise ValueError(
                f"Metrics and directions must have same length: "
                f"{len(metrics)} != {len(directions)}"
            )
        
        self.model = model
        self.search_space = search_space
        self.metrics = metrics
        self.directions = directions
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.study: optuna.Study = None
        
        logger.info(
            f"Multi-objective optimizer initialized: "
            f"metrics={metrics}, directions={directions}"
        )

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type")
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
        
        return params

    def _objective(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """
        Multi-objective function.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Tuple of metric values
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        # Create model with suggested parameters
        model = clone(self.model).set_params(**params)
        
        # Collect metrics
        metric_values = []
        
        for metric_name in self.metrics:
            if metric_name == "r2":
                # Cross-validation R² score
                cv_scores = cross_val_score(
                    model, self.X, self.y, cv=self.cv, scoring="r2", n_jobs=-1
                )
                metric_values.append(float(cv_scores.mean()))
            
            elif metric_name == "training_time":
                # Measure training time
                start_time = time.time()
                model.fit(self.X, self.y)
                training_time = time.time() - start_time
                metric_values.append(training_time)
            
            elif metric_name == "model_size":
                # Estimate model size (simplified)
                model.fit(self.X, self.y)
                
                # For tree-based models, use number of trees
                if hasattr(model, "n_estimators"):
                    size = model.n_estimators * 0.01  # MB estimate
                else:
                    size = 1.0  # Default 1MB
                
                metric_values.append(size)
            
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
        
        return tuple(metric_values)

    def search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            X: Training features
            y: Training targets
        
        Returns:
            Dictionary containing:
                - pareto_trials: List of Pareto-optimal trials
                - n_pareto_solutions: Number of Pareto solutions
                - study: Optuna study object
        
        Example:
            >>> results = optimizer.search(X_train, y_train)
            >>> print(f"Found {results['n_pareto_solutions']} Pareto solutions")
            Found 12 Pareto solutions
        """
        self.X = X
        self.y = y
        
        logger.info("Creating multi-objective study")
        
        try:
            # Create study with multiple objectives
            self.study = optuna.create_study(
                directions=self.directions,
                sampler=optuna.samplers.NSGAIISampler(seed=self.random_state),
            )
            
            # Optimize
            logger.info(f"Starting multi-objective optimization: {self.n_trials} trials")
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
            
            # Extract Pareto front
            pareto_trials = self.study.best_trials
            
            logger.info(
                f"Multi-objective optimization complete: "
                f"{len(pareto_trials)} Pareto solutions found"
            )
            
            return {
                "pareto_trials": pareto_trials,
                "n_pareto_solutions": len(pareto_trials),
                "study": self.study,
            }
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}", exc_info=True)
            raise

    def visualize_pareto_front(
        self, save_path: Path, metric_indices: Tuple[int, int] = (0, 1)
    ) -> None:
        """
        Visualize Pareto front.
        
        Args:
            save_path: Path to save visualization
            metric_indices: Indices of metrics to plot (2D)
        """
        if self.study is None:
            raise ValueError("No study available. Run search() first.")
        
        import plotly.graph_objects as go
        
        trials = self.study.best_trials
        
        if len(trials) == 0:
            logger.warning("No Pareto solutions to visualize")
            return
        
        # Extract metric values
        x_values = [t.values[metric_indices[0]] for t in trials]
        y_values = [t.values[metric_indices[1]] for t in trials]
        
        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="Pareto Solutions",
            )
        )
        
        fig.update_layout(
            title="Pareto Front",
            xaxis_title=self.metrics[metric_indices[0]],
            yaxis_title=self.metrics[metric_indices[1]],
        )
        
        fig.write_html(str(save_path))
        logger.info(f"Pareto front visualization saved to {save_path}")