#!/usr/bin/env python3
"""
Hashtag analysis CLI.

Command-line interface for hashtag intelligence.
"""

import asyncio
import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


@click.group()
def cli() -> None:
    """Hashtag Intelligence CLI"""
    pass


@cli.command()
@click.option("--hashtag", "-t", required=True, help="Hashtag to analyze")
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    help="Platform name",
)
@click.option("--user-id", "-u", help="User ID (optional)")
def analyze(hashtag: str, platform: str, user_id: str | None) -> None:
    """Analyze hashtag performance"""

    async def run():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        service = HashtagIntelligenceService(db_session=session)

        click.echo(f"Analyzing #{hashtag} on {platform}...\n")

        analysis = await service.analyze_hashtag(
            hashtag=hashtag,
            platform=platform,
            user_id=user_id,
        )

        # Display results
        click.echo("=" * 60)
        click.echo(f"ANALYSIS: #{hashtag}")
        click.echo("=" * 60)

        perf = analysis["performance"]
        click.echo("\nPerformance:")
        click.echo(f"  Total Uses: {perf['total_uses']}")
        click.echo(f"  Avg Engagement: {perf['avg_engagement']:.1f}")
        click.echo(f"  Engagement Lift: {perf['engagement_lift']:.1%}")
        click.echo(f"  Trend: {perf['trend_direction']}")
        click.echo(f"  ROI: {perf['roi']:.2f} per character")

        risk = analysis["risk"]
        click.echo("\nRisk Assessment:")
        click.echo(f"  Level: {risk['risk_level']}")
        click.echo(f"  Safe: {'✓' if risk['is_safe'] else '✗'}")
        click.echo(f"  Recommendation: {risk['recommendation']}")

        if risk["reasons"]:
            click.echo("  Reasons:")
            for reason in risk["reasons"]:
                click.echo(f"    - {reason}")

        related = analysis["related"]
        if related["synonyms"]:
            click.echo("\nSynonyms:")
            for syn in related["synonyms"][:5]:
                click.echo(f"  #{syn['hashtag']} ({syn['score']:.2f})")

    asyncio.run(run())


@cli.command()
@click.option("--content", "-c", required=True, help="Content text")
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    help="Platform name",
)
@click.option("--count", "-n", default=5, help="Number of recommendations")
def recommend(content: str, platform: str, count: int) -> None:
    """Get hashtag recommendations"""

    async def run():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        service = HashtagIntelligenceService(db_session=session)

        click.echo(f"Getting {count} recommendations for {platform}...\n")

        recommendations = await service.recommend_hashtags(
            content=content,
            platform=platform,
            count=count,
        )

        click.echo("=" * 60)
        click.echo("RECOMMENDATIONS")
        click.echo("=" * 60)

        for i, hashtag in enumerate(recommendations, 1):
            click.echo(f"{i}. #{hashtag}")

    asyncio.run(run())


@cli.command()
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    help="Platform name",
)
@click.option("--category", "-c", help="Category filter (optional)")
@click.option("--limit", "-l", default=20, help="Maximum results")
def trending(platform: str, category: str | None, limit: int) -> None:
    """Get trending hashtags"""

    async def run():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        service = HashtagIntelligenceService(db_session=session)

        click.echo(f"Getting trending hashtags on {platform}...\n")

        trending_list = await service.get_trending(
            platform=platform,
            category=category,
            limit=limit,
        )

        click.echo("=" * 60)
        click.echo(f"TRENDING - {platform.upper()}")
        click.echo("=" * 60)

        for i, trend in enumerate(trending_list, 1):
            click.echo(f"\n{i}. #{trend.hashtag}")
            click.echo(f"   Stage: {trend.stage.value}")
            click.echo(f"   Momentum: {trend.momentum_score:.1f}/100")
            click.echo(f"   Volume: {trend.current_volume:,}")
            click.echo(f"   Recommendation: {trend.recommendation}")

    asyncio.run(run())


@cli.command()
@click.option("--seed", "-s", required=True, help="Seed hashtag")
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    help="Platform name",
)
def discover(seed: str, platform: str) -> None:
    """Discover related hashtags"""

    async def run():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        service = HashtagIntelligenceService(db_session=session)

        click.echo(f"Discovering hashtags related to #{seed}...\n")

        discovery = await service.discovery_engine.discover(
            seed_hashtag=seed,
            platform=platform,
            include_trending=False,
        )

        click.echo("=" * 60)
        click.echo(f"DISCOVERY: #{seed}")
        click.echo("=" * 60)

        if discovery.synonyms:
            click.echo("\nSynonyms:")
            for ht in discovery.synonyms[:5]:
                click.echo(f"  #{ht.hashtag} ({ht.similarity_score:.2f})")

        if discovery.related:
            click.echo("\nRelated:")
            for ht in discovery.related[:5]:
                click.echo(f"  #{ht.hashtag} ({ht.similarity_score:.2f})")

        if discovery.niche_hashtags:
            click.echo("\nNiche Opportunities:")
            for ht in discovery.niche_hashtags[:5]:
                click.echo(
                    f"  #{ht.hashtag} (effectiveness: {ht.effectiveness_score:.1f})"
                )

    asyncio.run(run())


@cli.command()
@click.option("--hashtags", "-t", required=True, help="Comma-separated hashtags")
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["linkedin", "twitter", "bluesky"]),
    help="Platform name",
)
def validate(hashtags: str, platform: str) -> None:
    """Validate hashtag safety"""

    async def run():
        engine = create_engine("sqlite:///./bufferiq.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        service = HashtagIntelligenceService(db_session=session)

        hashtag_list = [h.strip() for h in hashtags.split(",")]

        click.echo(f"Validating {len(hashtag_list)} hashtags...\n")

        validation = await service.validate_hashtags(
            hashtags=hashtag_list,
            platform=platform,
        )

        click.echo("=" * 60)
        click.echo("VALIDATION RESULTS")
        click.echo("=" * 60)

        for hashtag, risk in validation.items():
            is_safe = risk.risk_level in ["none", "low"]
            icon = "✓" if is_safe else "✗"

            click.echo(f"\n{icon} #{hashtag}")
            click.echo(f"   Risk: {risk.risk_level}")
            click.echo(f"   Recommendation: {risk.recommendation}")

            if risk.risk_reasons:
                for reason in risk.risk_reasons:
                    click.echo(f"   - {reason}")

    asyncio.run(run())


if __name__ == "__main__":
    cli()