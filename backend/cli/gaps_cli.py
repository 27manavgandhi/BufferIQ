"""CLI tool for gap analysis."""

import asyncio
import click
import json
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService


@click.group()
def cli():
    """Gap Analysis CLI."""
    pass


@cli.command()
@click.option("--user-id", required=True, help="User identifier")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--competitors", help="Comma-separated competitor IDs")
@click.option("--industry", help="Industry category")
@click.option("--days", default=90, help="Lookback days")
def analyze(user_id, platform, competitors, industry, days):
    """Analyze content gaps."""
    click.echo(f"Analyzing gaps for {user_id} on {platform}...")
    
    # Parse competitors
    competitor_ids = competitors.split(",") if competitors else None
    
    # Run analysis
    result = asyncio.run(_analyze(user_id, platform, competitor_ids, industry, days))
    
    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"Gap Analysis Report")
    click.echo(f"{'='*60}")
    click.echo(f"Coverage Score: {result['coverage_score']:.1f}%")
    click.echo(f"Total Gaps: {result['total_gaps']}")
    click.echo(f"Critical Gaps: {len(result['critical_gaps'])}")
    click.echo(f"Important Gaps: {len(result['important_gaps'])}")
    click.echo(f"Recommendations: {result['recommendations_count']}")
    
    if result['critical_gaps']:
        click.echo(f"\n{'='*60}")
        click.echo("Critical Gaps:")
        for gap in result['critical_gaps'][:5]:
            click.echo(f"\n  - {gap['topic']}")
            click.echo(f"    Priority: {gap['priority_score']:.1f}")
            click.echo(f"    Competitors covering: {gap['competitor_coverage']}")


@cli.command()
@click.option("--user-id", required=True, help="User identifier")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--weeks", default=4, help="Number of weeks")
@click.option("--posts-per-week", default=3, help="Posts per week")
def calendar(user_id, platform, weeks, posts_per_week):
    """Generate content calendar."""
    click.echo(f"Generating {weeks}-week calendar for {user_id}...")
    
    result = asyncio.run(_generate_calendar(user_id, platform, weeks, posts_per_week))
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Content Calendar")
    click.echo(f"{'='*60}")
    click.echo(f"Total Pieces: {result['total_pieces']}")
    click.echo(f"Posting Frequency: {result['posting_frequency']} per week")
    
    click.echo(f"\nSchedule:")
    for item in result['calendar_items'][:10]:
        click.echo(f"  {item['date']}: {item['title']}")


@cli.command()
@click.option("--user-id", required=True, help="User identifier")
@click.option("--competitors", required=True, help="Comma-separated competitor IDs")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
def benchmark(user_id, competitors, platform):
    """Benchmark against competitors."""
    click.echo(f"Benchmarking {user_id} against competitors...")
    
    competitor_ids = competitors.split(",")
    result = asyncio.run(_benchmark(user_id, competitor_ids, platform))
    
    analysis = result['competitive_analysis']
    click.echo(f"\n{'='*60}")
    click.echo(f"Competitive Benchmark")
    click.echo(f"{'='*60}")
    click.echo(f"Your Rank: {analysis['user_rank']}/{len(analysis['competitor_profiles'])+1}")
    click.echo(f"Share of Voice: {analysis['share_of_voice']:.1f}%")
    click.echo(f"Engagement vs Avg: {analysis['engagement_vs_avg']:.2f}x")


async def _analyze(user_id, platform, competitor_ids, industry, days):
    """Run gap analysis."""
    # Create DB session
    engine = create_engine("sqlite:///./test.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    service = GapIntelligenceService(db_session=session)
    
    result = await service.analyze_gaps(
        user_id=user_id,
        platform=platform,
        competitor_ids=competitor_ids,
        industry=industry,
        lookback_days=days,
        include_recommendations=True
    )
    
    return result


async def _generate_calendar(user_id, platform, weeks, posts_per_week):
    """Generate calendar."""
    engine = create_engine("sqlite:///./test.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    service = GapIntelligenceService(db_session=session)
    
    result = await service.generate_calendar(
        user_id=user_id,
        platform=platform,
        weeks=weeks,
        posts_per_week=posts_per_week
    )
    
    return result


async def _benchmark(user_id, competitor_ids, platform):
    """Benchmark competitors."""
    engine = create_engine("sqlite:///./test.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    service = GapIntelligenceService(db_session=session)
    
    result = await service.benchmark_competitors(
        user_id=user_id,
        competitor_ids=competitor_ids,
        platform=platform
    )
    
    return result


if __name__ == "__main__":
    cli()