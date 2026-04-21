"""CLI commands for hyperparameter optimization."""

import asyncio
from pathlib import Path

import click

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.config_schema import OptimizationConfig
from bufferiq.ml.optimization.pipeline import OptimizationPipeline

logger = get_logger(__name__)


@click.group()
def optimize():
    """Hyperparameter optimization commands."""
    pass


@optimize.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to optimization config YAML file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config without running optimization",
)
def run(config: Path, dry_run: bool):
    """
    Run hyperparameter optimization.
    
    Example:
        python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml
    """
    try:
        # Load configuration
        opt_config = OptimizationConfig.from_yaml(config)
        logger.info(f"Loaded config from {config}")
        logger.info(f"Model: {opt_config.model_type}, Strategy: {opt_config.strategy}")
        
        if dry_run:
            click.echo("✓ Configuration valid")
            click.echo(f"  Model type: {opt_config.model_type}")
            click.echo(f"  Strategy: {opt_config.strategy}")
            click.echo(f"  CV folds: {opt_config.cv_folds}")
            if opt_config.n_iter:
                click.echo(f"  Iterations: {opt_config.n_iter}")
            return
        
        # Create pipeline
        pipeline = OptimizationPipeline(opt_config)
        
        # Run optimization
        click.echo("Starting optimization...")
        results = asyncio.run(pipeline.run())
        
        # Display results
        click.echo("\n" + "=" * 80)
        click.echo("OPTIMIZATION COMPLETE")
        click.echo("=" * 80)
        click.echo(f"Best score: {results['best_score']:.4f}")
        click.echo(f"Best params: {results['best_params']}")
        click.echo(f"Total trials: {results['total_trials']}")
        click.echo(f"Time taken: {results['optimization_time']:.2f}s")
        click.echo(f"Results saved to: {results['report_path']}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@optimize.command()
@click.option(
    "--optimization-id",
    type=str,
    required=True,
    help="Optimization ID (directory name)",
)
def best_params(optimization_id: str):
    """
    Show best parameters from an optimization.
    
    Example:
        python -m bufferiq.cli.optimize best-params --optimization-id xgboost_grid_20260412
    """
    import yaml
    
    params_file = Path(f"outputs/optimizations/{optimization_id}/best_params.yaml")
    
    if not params_file.exists():
        click.echo(f"Error: {params_file} not found", err=True)
        raise click.Abort()
    
    with open(params_file) as f:
        data = yaml.safe_load(f)
    
    click.echo("=" * 80)
    click.echo("BEST PARAMETERS")
    click.echo("=" * 80)
    click.echo(f"Best score: {data['best_score']:.4f}")
    click.echo(f"Trial ID: {data['trial_id']}")
    click.echo(f"\nParameters:")
    for param, value in data['best_params'].items():
        click.echo(f"  {param}: {value}")
    
    if 'baseline_score' in data:
        click.echo(f"\nBaseline score: {data['baseline_score']:.4f}")
        click.echo(f"Improvement: {data['improvement_pct']:.2f}%")


@optimize.command()
@click.argument(
    "optimization_dirs",
    type=click.Path(exists=True, path_type=Path),
    nargs=-1,
)
def compare(optimization_dirs: tuple):
    """
    Compare multiple optimization results.
    
    Example:
        python -m bufferiq.cli.optimize compare outputs/optimizations/opt1 outputs/optimizations/opt2
    """
    import json
    
    if len(optimization_dirs) < 2:
        click.echo("Error: Provide at least 2 optimization directories", err=True)
        raise click.Abort()
    
    results = []
    for opt_dir in optimization_dirs:
        report_file = Path(opt_dir) / "optimization_report.json"
        if not report_file.exists():
            click.echo(f"Warning: {report_file} not found, skipping", err=True)
            continue
        
        with open(report_file) as f:
            data = json.load(f)
            data['directory'] = opt_dir.name
            results.append(data)
    
    if not results:
        click.echo("Error: No valid optimization results found", err=True)
        raise click.Abort()
    
    # Sort by best score
    results.sort(key=lambda x: x.get('best_score', 0), reverse=True)
    
    click.echo("=" * 80)
    click.echo("OPTIMIZATION COMPARISON")
    click.echo("=" * 80)
    
    for i, result in enumerate(results, 1):
        click.echo(f"\n{i}. {result['directory']}")
        click.echo(f"   Best score: {result.get('best_score', 'N/A'):.4f}")
        click.echo(f"   Total trials: {result.get('total_trials', 'N/A')}")
        click.echo(f"   Duration: {result.get('total_duration', 0):.2f}s")


if __name__ == "__main__":
    optimize()