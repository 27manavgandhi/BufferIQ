"""Command-line interface for segmentation."""

import asyncio
import click
import json
from pathlib import Path
from datetime import datetime

from bufferiq.ml.segmentation.intelligence.service import SegmentationIntelligenceService
from bufferiq.ml.segmentation.intelligence.analyzer import SegmentationAnalyzer
from bufferiq.ml.segmentation.types import AudienceDataPoint


@click.group()
def cli():
    """BufferIQ Audience Segmentation CLI."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--platform",
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    default="linkedin",
    help="Platform type",
)
@click.option(
    "--output",
    type=click.Path(),
    default="outputs/segmentation",
    help="Output directory",
)
def segment(input_file: str, platform: str, output: str) -> None:
    """
    Segment audience into clusters with personas.

    INPUT_FILE: Path to JSON file with audience data
    """
    click.echo(f"🚀 Starting segmentation for {platform}...")

    # Load data
    with open(input_file) as f:
        data_dict = json.load(f)

    audience_data = [
        AudienceDataPoint(**item) for item in data_dict.get("audience_data", [])
    ]

    if not audience_data:
        click.echo("❌ No audience data found", err=True)
        return

    click.echo(f"📊 Loaded {len(audience_data)} audience members")

    # Run segmentation
    async def run():
        service = SegmentationIntelligenceService()
        return await service.segment_audience(audience_data, platform)

    result = asyncio.run(run())

    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = (
        output_path / f"segmentation_{platform}_{datetime.now().isoformat()}.json"
    )
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print results
    click.echo(f"✅ Segmentation complete!")
    click.echo(f"📁 Results saved to: {result_file}")
    click.echo(f"\n📈 Summary:")
    click.echo(f"  Segments: {result['n_segments']}")
    click.echo(f"  Algorithm: {result['clustering_algorithm']}")
    click.echo(f"  Quality (Silhouette): {result['clustering_quality']['silhouette_score']:.3f}")


@cli.command()
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default="outputs/analysis.json")
def analyze(result_file: str, output: str) -> None:
    """Analyze segmentation quality."""
    click.echo("🔍 Analyzing segmentation...")

    with open(result_file) as f:
        result = json.load(f)

    analyzer = SegmentationAnalyzer()
    personas = [
        type("Persona", (), p)() for p in result.get("personas", [])
    ]

    analysis = analyzer.analyze_results(personas)

    # Save analysis
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    click.echo(f"✅ Analysis complete!")
    click.echo(f"📁 Saved to: {output_path}")


@cli.command()
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--limit", type=int, default=5, help="Number of personas to show")
def personas(result_file: str, limit: int) -> None:
    """Show generated personas."""
    click.echo("👥 Generated Personas:\n")

    with open(result_file) as f:
        result = json.load(f)

    for i, persona in enumerate(result.get("personas", [])[:limit], 1):
        click.echo(f"{i}. {persona['persona_name']}")
        click.echo(f"   Size: {persona['size']} ({persona['size_percentage']:.1f}%)")
        click.echo(f"   Engagement: {persona['avg_engagement_rate']:.1%}")
        click.echo(f"   Topics: {', '.join(persona['primary_topics'][:3])}")
        click.echo()


@cli.command()
def version() -> None:
    """Show version."""
    click.echo("BufferIQ Day 22 - Audience Segmentation v1.0.0")


if __name__ == "__main__":
    cli()