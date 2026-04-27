"""CLI commands for ensemble models."""

import json
from pathlib import Path
from typing import Optional

import click
import joblib
import numpy as np
import yaml
from sklearn.metrics import r2_score

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.blending import BlendingEnsemble
from bufferiq.ml.ensemble.diversity_analyzer import DiversityAnalyzer
from bufferiq.ml.ensemble.ensemble_builder import EnsembleBuilder
from bufferiq.ml.ensemble.model_selector import ModelSelector
from bufferiq.ml.ensemble.performance_comparator import (
    EnsemblePerformanceComparator,
)
from bufferiq.ml.ensemble.stacking import StackingEnsemble
from bufferiq.ml.ensemble.voting import VotingEnsemble
from bufferiq.ml.ensemble.weight_optimizer import WeightOptimizer
from bufferiq.ml.ensemble.weighted_average import WeightedAverageEnsemble

logger = get_logger(__name__)


@click.group()
def ensemble():
    """Ensemble modeling commands."""
    pass


@ensemble.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to ensemble configuration YAML",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data (NPZ file with X_train, y_train)",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to validation data (NPZ file with X_val, y_val)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="outputs/models/ensembles",
    help="Output directory for ensemble",
)
def build(config: str, train_data: str, val_data: str, output_dir: str):
    """Build ensemble from configuration."""
    logger.info("Building ensemble from configuration")
    
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration: {cfg['model_name']}")
    
    # Load data
    train_npz = np.load(train_data)
    X_train = train_npz["X_train"]
    y_train = train_npz["y_train"]
    
    val_npz = np.load(val_data)
    X_val = val_npz["X_val"]
    y_val = val_npz["y_val"]
    
    logger.info(
        f"Loaded data: train={X_train.shape}, val={X_val.shape}"
    )
    
    # Build ensemble
    builder = EnsembleBuilder(
        model_paths=cfg["base_models"],
        ensemble_type=cfg["ensemble_type"],
        min_performance=cfg.get("selection", {}).get("min_performance", 0.70),
        min_diversity=cfg.get("selection", {}).get("min_diversity", 0.10),
        max_models=cfg.get("selection", {}).get("max_models", 5),
        weight_optimization=cfg.get("weight_optimization", "performance"),
        output_dir=Path(output_dir),
    )
    
    ensemble = builder.build(X_train, y_train, X_val, y_val)
    
    # Evaluate
    pred_val = ensemble.predict(X_val)
    r2_val = r2_score(y_val, pred_val)
    
    logger.info(f"Validation R²: {r2_val:.4f}")
    
    # Save ensemble
    output_path = Path(output_dir) / f"{cfg['model_name']}.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_path)
    
    logger.info(f"Ensemble saved to {output_path}")
    
    click.echo(f"✓ Ensemble built successfully: R² = {r2_val:.4f}")
    click.echo(f"✓ Saved to: {output_path}")


@ensemble.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to base models (can specify multiple)",
)
@click.option(
    "--weights",
    type=str,
    help="Comma-separated weights (e.g., '0.5,0.3,0.2')",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for ensemble",
)
def voting(models: tuple, weights: Optional[str], train_data: str, output: str):
    """Create voting ensemble."""
    logger.info("Creating voting ensemble")
    
    # Load models
    base_models = [joblib.load(path) for path in models]
    logger.info(f"Loaded {len(base_models)} base models")
    
    # Parse weights
    weights_array = None
    if weights:
        weights_array = np.array([float(w) for w in weights.split(",")])
        logger.info(f"Using custom weights: {weights_array}")
    
    # Load data
    data = np.load(train_data)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Create ensemble
    ensemble = VotingEnsemble(base_models, weights=weights_array)
    ensemble.fit(X_train, y_train)
    
    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_path)
    
    logger.info(f"Voting ensemble saved to {output_path}")
    click.echo(f"✓ Voting ensemble created and saved to {output_path}")


@ensemble.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to base models",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data",
)
@click.option(
    "--cv",
    type=int,
    default=5,
    help="Cross-validation folds",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for ensemble",
)
def stacking(models: tuple, train_data: str, cv: int, output: str):
    """Create stacking ensemble."""
    from sklearn.linear_model import Ridge
    
    logger.info("Creating stacking ensemble")
    
    # Load models
    base_models = [joblib.load(path) for path in models]
    logger.info(f"Loaded {len(base_models)} base models")
    
    # Load data
    data = np.load(train_data)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Create ensemble
    meta_learner = Ridge(alpha=1.0)
    ensemble = StackingEnsemble(base_models, meta_learner, cv=cv)
    ensemble.fit(X_train, y_train)
    
    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_path)
    
    logger.info(f"Stacking ensemble saved to {output_path}")
    click.echo(f"✓ Stacking ensemble created and saved to {output_path}")


@ensemble.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to base models",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data",
)
@click.option(
    "--method",
    type=click.Choice(["uniform", "performance", "optuna", "grid"]),
    default="optuna",
    help="Weight optimization method",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Number of Optuna trials",
)
def optimize_weights(
    models: tuple, train_data: str, method: str, n_trials: int
):
    """Optimize ensemble weights."""
    logger.info(f"Optimizing weights using {method}")
    
    # Load models
    base_models = [joblib.load(path) for path in models]
    logger.info(f"Loaded {len(base_models)} base models")
    
    # Load data
    data = np.load(train_data)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Optimize weights
    optimizer = WeightOptimizer(base_models, method=method, n_trials=n_trials)
    result = optimizer.optimize_with_details(X_train, y_train)
    
    logger.info(f"Optimal weights: {result['weights']}")
    
    if "best_score" in result:
        logger.info(f"Best score: {result['best_score']:.4f}")
    
    # Display results
    click.echo("\n=== Weight Optimization Results ===")
    click.echo(f"Method: {method}")
    click.echo(f"Optimal weights: {result['weights']}")
    if "best_score" in result:
        click.echo(f"Best R²: {result['best_score']:.4f}")


@ensemble.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to base models",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to validation data",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="outputs/ensembles/diversity",
    help="Output directory for visualizations",
)
def analyze_diversity(models: tuple, val_data: str, output_dir: str):
    """Analyze diversity of base models."""
    logger.info("Analyzing model diversity")
    
    # Load models
    base_models = [joblib.load(path) for path in models]
    logger.info(f"Loaded {len(base_models)} base models")
    
    # Load data
    data = np.load(val_data)
    X_val = data["X_val"]
    y_val = data["y_val"]
    
    # Get predictions
    predictions = np.column_stack([
        model.predict(X_val) for model in base_models
    ])
    
    # Analyze diversity
    model_names = [f"Model_{i+1}" for i in range(len(base_models))]
    output_path = Path(output_dir)
    
    metrics = DiversityAnalyzer.analyze_all(
        predictions, y_val, model_names, output_path
    )
    
    # Display results
    click.echo("\n=== Diversity Analysis ===")
    click.echo(f"Correlation diversity: {metrics['correlation_diversity']:.4f}")
    click.echo(f"Disagreement diversity: {metrics['disagreement_diversity']:.4f}")
    click.echo(f"Avg Q-statistic: {metrics['avg_q_statistic']:.4f}")
    click.echo(f"\nVisualizations saved to: {output_path}")


@ensemble.command()
@click.option(
    "--ensemble",
    type=click.Path(exists=True),
    required=True,
    help="Path to ensemble model",
)
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to base models",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to test data",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="outputs/ensembles/comparison",
    help="Output directory",
)
def compare(ensemble: str, models: tuple, test_data: str, output_dir: str):
    """Compare ensemble against base models."""
    logger.info("Comparing ensemble performance")
    
    # Load ensemble
    ensemble_model = joblib.load(ensemble)
    logger.info(f"Loaded ensemble from {ensemble}")
    
    # Load base models
    base_models = [joblib.load(path) for path in models]
    model_names = [f"Model_{i+1}" for i in range(len(base_models))]
    logger.info(f"Loaded {len(base_models)} base models")
    
    # Load data
    data = np.load(test_data)
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Compare
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(
        ensemble_model, base_models, X_test, y_test, model_names
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparator.visualize_comparison(
        results, output_path / "comparison.png"
    )
    comparator.export_report(results, output_path / "report.json")
    
    # Display results
    click.echo("\n=== Performance Comparison ===")
    click.echo(f"Ensemble R²: {results['ensemble_metrics']['r2']:.4f}")
    click.echo(f"Improvement: {results['improvement_pct']:.2f}%")
    click.echo(f"\nResults saved to: {output_path}")


@ensemble.command()
@click.option(
    "--models",
    multiple=True,
    required=True,
    help="Paths to candidate models",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to validation data",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for best ensemble",
)
def auto(models: tuple, train_data: str, val_data: str, output: str):
    """Automatically select and build best ensemble."""
    logger.info("Auto-building best ensemble")
    
    # Load data
    train_npz = np.load(train_data)
    X_train = train_npz["X_train"]
    y_train = train_npz["y_train"]
    
    val_npz = np.load(val_data)
    X_val = val_npz["X_val"]
    y_val = val_npz["y_val"]
    
    # Build ensemble
    builder = EnsembleBuilder(
        model_paths=list(models),
        ensemble_type="auto",
    )
    
    ensemble = builder.build(X_train, y_train, X_val, y_val)
    
    # Evaluate
    pred_val = ensemble.predict(X_val)
    r2_val = r2_score(y_val, pred_val)
    
    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_path)
    
    click.echo(f"✓ Best ensemble: R² = {r2_val:.4f}")
    click.echo(f"✓ Saved to: {output_path}")