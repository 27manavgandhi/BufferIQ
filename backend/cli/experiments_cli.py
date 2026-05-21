"""
Experiments CLI tool.

Example usage:
    python cli/experiments_cli.py create --name "Test" --platform linkedin
    python cli/experiments_cli.py analyze --experiment-id exp_001
    python cli/experiments_cli.py list
"""

import click
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService
from bufferiq.ml.experiments.design.designer import Variant, MetricType


@click.group()
def cli():
    """Experiments CLI."""
    pass


@cli.command()
@click.option("--name", required=True, help="Experiment name")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--baseline-rate", type=float, required=True)
@click.option("--mde", type=float, default=0.10)
def create(name, platform, baseline_rate, mde):
    """Create experiment."""
    
    async def _create():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        service = ExperimentIntelligenceService(session)
        
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {"version": "new"})
        ]
        
        exp = await service.create_experiment(
            name=name,
            description=f"Test on {platform}",
            variants=variants,
            platform=platform,
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=baseline_rate,
            mde=mde
        )
        
        click.echo(f"✓ Created: {exp.experiment_id}")
    
    asyncio.run(_create())


@cli.command()
@click.option("--experiment-id", required=True)
def analyze(experiment_id):
    """Analyze experiment."""
    
    async def _analyze():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        service = ExperimentIntelligenceService(session)
        results = await service.analyze_experiment(experiment_id=experiment_id)
        
        if results["status"] == "complete":
            click.echo(f"Winner: {results['winner_variant']}")
            click.echo(f"P-value: {results['statistical_result']['p_value']:.4f}")
        else:
            click.echo(f"Status: {results['status']}")
    
    asyncio.run(_analyze())


@cli.command()
def list():
    """List experiments."""
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    service = ExperimentIntelligenceService(session)
    experiments = service.list_experiments()
    
    for exp in experiments:
        click.echo(f"{exp.experiment_id}: {exp.name} ({exp.platform})")


if __name__ == "__main__":
    cli()