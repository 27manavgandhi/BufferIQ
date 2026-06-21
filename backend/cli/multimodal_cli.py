#!/usr/bin/env python3
"""
CLI tool for multi-modal analysis.

Usage:
    python -m cli.multimodal_cli image analyze --path image.jpg --platform linkedin
    python -m cli.multimodal_cli video analyze --url video.mp4 --platform twitter
    python -m cli.multimodal_cli link analyze --url https://example.com --platform bluesky
"""

import click
import asyncio
import json
from pathlib import Path

from bufferiq.ml.multimodal.images.analyzer import ImageAnalyzer
from bufferiq.ml.multimodal.videos.analyzer import VideoAnalyzer
from bufferiq.ml.multimodal.links.analyzer import LinkPreviewAnalyzer


@click.group()
def cli():
    """Multi-modal content analysis CLI."""
    pass


@cli.group()
def image():
    """Image analysis commands."""
    pass


@cli.group()
def video():
    """Video analysis commands."""
    pass


@cli.group()
def link():
    """Link preview commands."""
    pass


@image.command()
@click.option("--path", required=True, help="Image path or URL")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--output", help="Output JSON file")
def analyze_image(path: str, platform: str, output: str):
    """Analyze an image."""
    async def run():
        analyzer = ImageAnalyzer()
        result = await analyzer.analyze(path, platform)  # type: ignore
        
        click.echo(f"✅ Image analyzed successfully")
        click.echo(f"   Aesthetic score: {result.aesthetic_score:.1f}/100")
        click.echo(f"   Objects: {len(result.objects)}")
        click.echo(f"   Faces: {len(result.faces)}")
        
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2))
            click.echo(f"   Saved to: {output}")
        
        return result
    
    asyncio.run(run())


@video.command()
@click.option("--url", required=True, help="Video URL or path")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--output", help="Output JSON file")
def analyze_video(url: str, platform: str, output: str):
    """Analyze a video."""
    async def run():
        analyzer = VideoAnalyzer()
        result = await analyzer.analyze(url, platform)  # type: ignore
        
        click.echo(f"✅ Video analyzed successfully")
        click.echo(f"   Duration: {result.metadata.duration_seconds:.1f}s")
        click.echo(f"   Keyframes: {len(result.keyframes)}")
        click.echo(f"   Scenes: {len(result.scenes)}")
        
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2))
            click.echo(f"   Saved to: {output}")
        
        return result
    
    asyncio.run(run())


@link.command()
@click.option("--url", required=True, help="Link URL")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--output", help="Output JSON file")
def analyze_link(url: str, platform: str, output: str):
    """Analyze a link preview."""
    async def run():
        analyzer = LinkPreviewAnalyzer()
        result = await analyzer.analyze(url, platform)  # type: ignore
        
        click.echo(f"✅ Link analyzed successfully")
        click.echo(f"   Quality: {result.quality_scores.overall_quality:.1f}/100")
        click.echo(f"   CTR prediction: {result.ctr_prediction:.2%}")
        click.echo(f"   Suggestions: {len(result.optimization_suggestions)}")
        
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2))
            click.echo(f"   Saved to: {output}")
        
        return result
    
    asyncio.run(run())


if __name__ == "__main__":
    cli()