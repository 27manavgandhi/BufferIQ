"""
Content analysis CLI.

Command-line interface for content intelligence features.
"""

import click

from bufferiq.ml.content.intelligence.service import ContentIntelligenceService


@click.group()
def content() -> None:
    """Content intelligence commands."""
    pass


@content.command()
@click.option("--text", required=True, help="Text to analyze")
@click.option(
    "--platform",
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    default="linkedin",
    help="Platform type",
)
@click.option(
    "--no-optimization",
    is_flag=True,
    help="Disable optimization suggestions",
)
def analyze(text: str, platform: str, no_optimization: bool) -> None:
    """Analyze content text."""
    service = ContentIntelligenceService()

    try:
        result = service.analyze_content(
            text=text,
            platform=platform,
            include_optimization=not no_optimization,
        )

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("CONTENT ANALYSIS RESULTS")
        click.echo("=" * 60 + "\n")

        # Sentiment
        click.echo("Sentiment:")
        sentiment = result.get("sentiment", {})
        click.echo(
            f"  - {sentiment.get('sentiment', 'N/A')} "
            f"(confidence: {sentiment.get('confidence', 0):.2f})"
        )

        # Quality
        click.echo("\nQuality:")
        quality = result.get("quality", {})
        click.echo(f"  - Score: {quality.get('score', 0):.1f}/100")
        click.echo(
            f"  - Grammar errors: {quality.get('grammar_errors', 0)}"
        )
        click.echo(
            f"  - Spelling errors: {quality.get('spelling_errors', 0)}"
        )

        # Readability
        if "readability" in result:
            click.echo("\nReadability:")
            readability = result["readability"]
            click.echo(
                f"  - Difficulty: {readability.get('reading_difficulty', 'N/A')}"
            )
            click.echo(
                f"  - Grade level: {readability.get('average_grade_level', 0):.1f}"
            )

        # Optimization
        if "optimization" in result:
            click.echo("\nOptimization:")
            optimization = result["optimization"]
            click.echo(
                f"  - Overall score: {optimization.get('overall_score', 0):.1f}/100"
            )
            click.echo(
                f"  - Predicted lift: {optimization.get('predicted_engagement_lift', 0):.1f}%"
            )

            suggestions = optimization.get("suggestions", [])
            if suggestions:
                click.echo("\n  Suggestions:")
                for sug in suggestions[:5]:  # Top 5
                    click.echo(
                        f"    • [{sug['priority']}] {sug['type']}: {sug['impact']}"
                    )

        click.echo("\n" + "=" * 60 + "\n")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    content()