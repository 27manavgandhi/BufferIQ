"""CLI commands for model training."""

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from bufferiq.core.database import async_session_maker
from bufferiq.core.logging import get_logger
from bufferiq.ml.training.config_schema import TrainingPipelineConfig
from bufferiq.ml.training.model_registry import ModelRegistry
from bufferiq.ml.training.pipeline import TrainingPipeline

logger = get_logger(__name__)


@click.group()
def train() -> None:
    """Model training commands."""
    pass


@train.command()
@click.option(
    "--config", required=True, type=str, help="Path to training config YAML"
)
@click.option("--user-id", type=int, help="Optional user ID filter")
@click.option("--cv", is_flag=True, help="Use cross-validation")
def run(config: str, user_id: int | None, cv: bool) -> None:
    """Train model with config."""
    asyncio.run(_run_training(config, user_id, cv))


async def _run_training(
    config_path: str, user_id: int | None, use_cv: bool
) -> None:
    """Async training execution."""
    try:
        # Load config
        config = TrainingPipelineConfig.from_yaml(config_path)

        # Override CV if specified
        if use_cv:
            config.experiment.use_cross_validation = True

        # Create pipeline
        async with async_session_maker() as session:
            pipeline = TrainingPipeline(config, session)

            # Run training
            if config.experiment.use_cross_validation:
                results = await pipeline.run_with_cross_validation()
            else:
                results = await pipeline.run()

        # Print results
        click.echo("\n" + "=" * 80)
        click.echo("Training Complete!")
        click.echo("=" * 80)
        click.echo(f"Experiment: {results['experiment_name']}")
        click.echo(f"Experiment Dir: {results['experiment_dir']}")

        if "test_metrics" in results:
            click.echo(f"\nTest Metrics:")
            for metric, value in results["test_metrics"].items():
                click.echo(f"  {metric}: {value:.4f}")

        click.echo("\n✅ Success!")

    except Exception as e:
        click.echo(f"\n❌ Training failed: {e}", err=True)
        sys.exit(1)


@train.command()
def list_experiments() -> None:
    """List all experiments."""
    from bufferiq.ml.training.experiment_tracker import ExperimentTracker

    experiments = ExperimentTracker.list_experiments()

    if not experiments:
        click.echo("No experiments found.")
        return

    click.echo(f"\nFound {len(experiments)} experiments:\n")

    for exp in experiments:
        click.echo(f"  - {exp['experiment_name']}")
        click.echo(f"    Started: {exp['start_time']}")
        click.echo(f"    Duration: {exp.get('duration_seconds', 0):.2f}s")
        click.echo()


@train.command()
@click.option("--name", required=True, help="Experiment name")
def show_experiment(name: str) -> None:
    """Show experiment details."""
    from bufferiq.ml.training.experiment_tracker import ExperimentTracker

    experiments = ExperimentTracker.list_experiments()
    exp = next((e for e in experiments if e["experiment_name"] == name), None)

    if not exp:
        click.echo(f"Experiment '{name}' not found.")
        sys.exit(1)

    tracker = ExperimentTracker.load_experiment(exp["experiment_dir"])

    click.echo(f"\nExperiment: {name}")
    click.echo("=" * 80)
    click.echo(f"\nParameters:")
    for key, value in tracker.get_params().items():
        click.echo(f"  {key}: {value}")

    click.echo(f"\nMetrics:")
    for key, values in tracker.get_metrics().items():
        if values:
            click.echo(f"  {key}: {values[-1]['value']:.4f}")


@train.command()
def list_models() -> None:
    """List registered models."""
    registry = ModelRegistry()
    models = registry.list_models()

    if not models:
        click.echo("No models registered.")
        return

    click.echo(f"\nFound {len(models)} registered models:\n")

    for model in models:
        click.echo(f"  - {model['version']}")
        click.echo(f"    Model ID: {model['model_id']}")
        click.echo(f"    Production: {model.get('is_production', False)}")
        click.echo(f"    Registered: {model['registered_at']}")
        if "r2" in model.get("metrics", {}):
            click.echo(f"    R²: {model['metrics']['r2']:.4f}")
        click.echo()


@train.command()
@click.option("--version", required=True, help="Version to promote")
def promote(version: str) -> None:
    """Promote model to production."""
    registry = ModelRegistry()

    try:
        registry.promote_to_production(version)
        click.echo(f"\n✅ Promoted {version} to production")
    except ValueError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@train.command()
@click.option("--metric", default="r2", help="Metric to compare")
def best_model(metric: str) -> None:
    """Show best model by metric."""
    registry = ModelRegistry()

    try:
        best = registry.get_best_model(metric=metric)

        click.echo(f"\nBest model by {metric}:")
        click.echo(f"  Version: {best['version']}")
        click.echo(f"  {metric}: {best['metrics'][metric]:.4f}")
        click.echo(f"  Production: {best.get('is_production', False)}")

    except ValueError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    train()