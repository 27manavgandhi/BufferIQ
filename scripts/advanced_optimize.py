"""Standalone script for advanced Optuna optimization."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.optuna_optimizer import OptunaOptimizer
from bufferiq.ml.optimization.optuna_pruners import PrunerRegistry
from bufferiq.ml.optimization.optuna_samplers import SamplerRegistry
from bufferiq.ml.optimization.multi_objective import MultiObjectiveOptimizer
from bufferiq.ml.optimization.parallel_optimizer import ParallelOptimizer
from bufferiq.ml.optimization.param_importance import (
    HyperparameterImportanceAnalyzer,
)
from bufferiq.ml.optimization.advanced_visualizer import (
    AdvancedOptimizationVisualizer,
)

logger = get_logger(__name__)


def main():
    """Run advanced optimization."""
    parser = argparse.ArgumentParser(description="Advanced Optuna optimization")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to optimization config YAML",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="outputs/features/train_features.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["optuna", "multi-objective", "parallel"],
        default="optuna",
        help="Optimization mode",
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["engagement_score"]).values
    y = df["engagement_score"].values
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create model
    model_type = config["model_type"]
    if model_type == "xgboost":
        model = XGBRegressor(random_state=42)
    elif model_type == "lightgbm":
        model = LGBMRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run optimization based on mode
    if args.mode == "optuna":
        run_optuna_optimization(model, config, X, y)
    elif args.mode == "multi-objective":
        run_multi_objective_optimization(model, config, X, y)
    elif args.mode == "parallel":
        run_parallel_optimization(model, config, X, y)


def run_optuna_optimization(model, config, X, y):
    """Run standard Optuna optimization."""
    logger.info("Running Optuna optimization")
    
    # Create sampler and pruner
    sampler = SamplerRegistry.get_sampler(
        config["sampler"],
        seed=config.get("random_state", 42)
    )
    
    pruner = None
    if config.get("pruner"):
        pruner = PrunerRegistry.get_pruner(config["pruner"])
    
    # Run optimization
    optimizer = OptunaOptimizer(
        model=model,
        search_space=config["search_space"],
        n_trials=config["n_trials"],
        timeout=config.get("timeout"),
        sampler=sampler,
        pruner=pruner,
        direction=config["direction"],
        study_name=config["study_name"],
        storage=config.get("storage"),
        cv=config["cv_folds"],
        scoring=config["metric"],
        random_state=config.get("random_state", 42),
    )
    
    results = optimizer.search(X, y)
    
    logger.info(
        f"Optimization complete: "
        f"best_score={results['best_score']:.4f}, "
        f"n_trials={results['n_trials']}, "
        f"n_pruned={results['n_pruned']}"
    )
    
    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "best_params.yaml", "w") as f:
        yaml.dump(results["best_params"], f)
    
    # Create visualizations
    if config.get("create_visualizations", True):
        logger.info("Creating visualizations")
        visualizer = AdvancedOptimizationVisualizer(results["study"])
        visualizer.create_all_visualizations(output_dir / "visualizations")
    
    # Analyze importance
    logger.info("Analyzing hyperparameter importance")
    analyzer = HyperparameterImportanceAnalyzer(results["study"])
    importance = analyzer.calculate_importance()
    
    analyzer.visualize_importance(
        importance,
        output_dir / "param_importance.png",
    )
    analyzer.export_rankings(
        importance,
        output_dir / "param_rankings.json",
    )
    
    logger.info(f"Results saved to {output_dir}")


def run_multi_objective_optimization(model, config, X, y):
    """Run multi-objective optimization."""
    logger.info("Running multi-objective optimization")
    
    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=config["search_space"],
        metrics=config["metrics"],
        directions=config["directions"],
        n_trials=config["n_trials"],
        cv=config["cv_folds"],
        random_state=config.get("random_state", 42),
    )
    
    results = optimizer.search(X, y)
    
    logger.info(
        f"Multi-objective optimization complete: "
        f"{results['n_pareto_solutions']} Pareto solutions"
    )
    
    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize Pareto front
    optimizer.visualize_pareto_front(output_dir / "pareto_front.html")
    
    logger.info(f"Results saved to {output_dir}")


def run_parallel_optimization(model, config, X, y):
    """Run parallel optimization."""
    logger.info("Running parallel optimization")
    
    # Define objective
    def objective(trial):
        from sklearn.model_selection import cross_val_score
        
        params = {}
        for param_name, param_config in config["search_space"].items():
            if param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
        
        model_instance = model.__class__(**params)
        scores = cross_val_score(
            model_instance, X, y, cv=config["cv_folds"], scoring="r2"
        )
        return scores.mean()
    
    # Create sampler and pruner
    sampler = SamplerRegistry.get_sampler(
        config["sampler"],
        seed=config.get("random_state", 42)
    )
    
    pruner = None
    if config.get("pruner"):
        pruner = PrunerRegistry.get_pruner(config["pruner"])
    
    # Run parallel optimization
    parallel_opt = ParallelOptimizer(
        objective=objective,
        study_name=config["study_name"],
        storage=config["storage"],
        n_workers=config.get("n_workers", 4),
        n_trials_per_worker=config.get("n_trials_per_worker", 25),
    )
    
    study = parallel_opt.run(
        direction=config["direction"],
        sampler=sampler,
        pruner=pruner,
    )
    
    logger.info(
        f"Parallel optimization complete: "
        f"best_score={study.best_value:.4f}, "
        f"n_trials={len(study.trials)}"
    )
    
    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "best_params.yaml", "w") as f:
        yaml.dump(study.best_params, f)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()