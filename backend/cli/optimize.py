"""CLI commands for hyperparameter optimization."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
import yaml

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


@click.group()
def optimize() -> None:
    """Hyperparameter optimization commands."""
    pass


# =========================================================
# RUN (GRID / RANDOM / BASIC PIPELINE)
# =========================================================
@optimize.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to optimization config YAML file",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default="outputs/features/train_features.csv",
    help="Path to training data CSV",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config without running optimization",
)
def run(config: str, data_path: str, dry_run: bool) -> None:
    """
    Run hyperparameter optimization.

    Example:
        optimize run --config configs/optimization/xgboost_grid.yaml
    """
    import pandas as pd
    from bufferiq.ml.optimization.pipeline import OptimizationPipeline

    try:
        logger.info(f"Loading config from {config}")

        with open(config) as f:
            config_dict = yaml.safe_load(f)

        if dry_run:
            logger.info("Dry run - config validation only")
            logger.info(f"Strategy: {config_dict.get('strategy')}")
            logger.info(f"Model: {config_dict.get('model_type')}")
            return

        # Load data
        df = pd.read_csv(data_path)
        X = df.drop(columns=["engagement_score"]).values
        y = df["engagement_score"].values

        # Run optimization
        pipeline = OptimizationPipeline(config_dict)
        results = pipeline.run(X, y)

        click.echo("\n" + "=" * 80)
        click.echo("OPTIMIZATION COMPLETE")
        click.echo("=" * 80)
        click.echo(f"Best score: {results['best_score']:.4f}")
        click.echo(f"Best params: {results['best_params']}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise click.Abort()


# =========================================================
# OPTUNA
# =========================================================
@optimize.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default="outputs/features/train_features.csv",
)
@click.option("--dry-run", is_flag=True)
def optuna(config: str, data_path: str, dry_run: bool) -> None:
    """
    Run Optuna optimization.
    """
    import pandas as pd
    from bufferiq.ml.optimization.optuna_optimizer import OptunaOptimizer
    from bufferiq.ml.optimization.optuna_pruners import PrunerRegistry
    from bufferiq.ml.optimization.optuna_samplers import SamplerRegistry

    logger.info(f"Loading Optuna config from {config}")

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    if dry_run:
        logger.info("Dry run - config validation only")
        logger.info(f"Study: {config_dict.get('study_name')}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=["engagement_score"]).values
    y = df["engagement_score"].values

    # Model
    if config_dict["model_type"] == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=42)
    elif config_dict["model_type"] == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=42)
    else:
        raise ValueError("Unsupported model")

    sampler = SamplerRegistry.get_sampler(
        config_dict["sampler"],
        seed=config_dict.get("random_state", 42),
    )

    pruner = None
    if config_dict.get("pruner"):
        pruner = PrunerRegistry.get_pruner(config_dict["pruner"])

    optimizer = OptunaOptimizer(
        model=model,
        search_space=config_dict["search_space"],
        n_trials=config_dict["n_trials"],
        sampler=sampler,
        pruner=pruner,
        direction=config_dict["direction"],
        study_name=config_dict["study_name"],
        storage=config_dict.get("storage"),
        cv=config_dict["cv_folds"],
        scoring=config_dict["metric"],
    )

    results = optimizer.search(X, y)

    logger.info(f"Best score: {results['best_score']:.4f}")


# =========================================================
# MULTI OBJECTIVE
# =========================================================
@optimize.command(name="multi-objective")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default="outputs/features/train_features.csv",
)
def multi_objective(config: str, data_path: str) -> None:
    """
    Multi-objective optimization.
    """
    import pandas as pd
    from bufferiq.ml.optimization.multi_objective import MultiObjectiveOptimizer

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["engagement_score"]).values
    y = df["engagement_score"].values

    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=42)

    optimizer = MultiObjectiveOptimizer(
        model=model,
        search_space=config_dict["search_space"],
        metrics=config_dict["metrics"],
        directions=config_dict["directions"],
        n_trials=config_dict["n_trials"],
        cv=config_dict["cv_folds"],
    )

    results = optimizer.search(X, y)

    logger.info(f"Pareto solutions: {results['n_pareto_solutions']}")


# =========================================================
# RESUME STUDY
# =========================================================
@optimize.command()
@click.option("--study-name", required=True)
@click.option("--storage", default="sqlite:///outputs/optimizations/optuna.db")
@click.option("--n-trials", type=int, default=50)
def resume(study_name: str, storage: str, n_trials: int) -> None:
    from bufferiq.ml.optimization.study_manager import OptunaStudyManager

    manager = OptunaStudyManager(storage)
    study = manager.load_study(study_name)

    logger.info(f"Resuming {study_name} with {n_trials} trials")
    study.optimize(lambda trial: 0.0, n_trials=n_trials)


# =========================================================
# IMPORTANCE
# =========================================================
@optimize.command()
@click.option("--study-name", required=True)
@click.option("--storage", default="sqlite:///outputs/optimizations/optuna.db")
@click.option("--output-dir", default="outputs/importance")
def importance(study_name: str, storage: str, output_dir: str) -> None:
    from bufferiq.ml.optimization.study_manager import OptunaStudyManager
    from bufferiq.ml.optimization.param_importance import (
        HyperparameterImportanceAnalyzer,
    )

    manager = OptunaStudyManager(storage)
    study = manager.load_study(study_name)

    analyzer = HyperparameterImportanceAnalyzer(study)
    importance = analyzer.calculate_importance()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer.export_rankings(importance, output_path / "rankings.json")

    click.echo("\nTop parameters:")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        click.echo(f"{k}: {v:.4f}")


# =========================================================
# BEST PARAMS
# =========================================================
@optimize.command(name="best-params")
@click.option("--optimization-id", required=True)
@click.option("--output-dir", default="outputs/optimizations")
def best_params(optimization_id: str, output_dir: str) -> None:
    path = Path(output_dir) / "best_params" / f"{optimization_id}_best.yaml"

    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    with open(path) as f:
        params = yaml.safe_load(f)

    click.echo(yaml.dump(params))


# =========================================================
# COMPARE
# =========================================================
@optimize.command()
@click.argument("optimization_dirs", nargs=-1, type=click.Path(exists=True))
def compare(optimization_dirs: tuple) -> None:
    if len(optimization_dirs) < 2:
        logger.error("Need at least 2 dirs")
        return

    results = []

    for d in optimization_dirs:
        file = Path(d) / "results.json"
        if file.exists():
            with open(file) as f:
                results.append((d, json.load(f)))

    for d, r in results:
        click.echo(f"\n{d}")
        click.echo(f"Best score: {r.get('best_score')}")


# =========================================================
# LIST STUDIES
# =========================================================
@optimize.command(name="list-studies")
@click.option("--storage", default="sqlite:///outputs/optimizations/optuna.db")
def list_studies(storage: str) -> None:
    from bufferiq.ml.optimization.study_manager import OptunaStudyManager

    manager = OptunaStudyManager(storage)
    studies = manager.list_studies()

    for s in studies:
        click.echo(s)


if __name__ == "__main__":
    optimize()