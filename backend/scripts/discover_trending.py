#!/usr/bin/env python3
"""
Discover trending hashtags script.

Finds and reports on trending hashtags.
"""

import asyncio
import argparse
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


async def discover_trending(
    platform: str,
    category: str | None = None,
    limit: int = 20,
) -> None:
    """
    Discover trending hashtags.

    Args:
        platform: Platform name
        category: Optional category filter
        limit: Maximum results
    """
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize service
    service = HashtagIntelligenceService(db_session=session)

    print(f"Discovering trending hashtags on {platform}...")
    if category:
        print(f"Category: {category}")

    # Get trending
    trending = await service.get_trending(
        platform=platform,
        category=category,
        limit=limit,
    )

    print(f"\n✓ Found {len(trending)} trending hashtags\n")

    # Display results
    print("=" * 80)
    print(f"TRENDING HASHTAGS - {platform.upper()}")
    print("=" * 80)

    for i, trend in enumerate(trending, 1):
        print(f"\n{i}. #{trend.hashtag}")
        print(f"   Stage: {trend.stage.value}")
        print(f"   Momentum: {trend.momentum_score:.1f}/100")
        print(f"   Volume: {trend.current_volume:,}")
        print(f"   Change: {trend.volume_change:+.1%}")
        print(f"   Recommendation: {trend.recommendation}")

        if trend.related_topics:
            print(f"   Related: {', '.join(trend.related_topics[:3])}")

    # Generate report
    print("\n" + "=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)

    # Group by stage
    by_stage: dict = {}
    for trend in trending:
        stage = trend.stage.value
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(trend)

    print("\nBy Stage:")
    for stage, trends in sorted(by_stage.items()):
        print(f"  {stage.capitalize()}: {len(trends)}")

    # Top momentum
    print("\nTop Momentum:")
    top_momentum = sorted(trending, key=lambda t: t.momentum_score, reverse=True)[:5]
    for trend in top_momentum:
        print(f"  #{trend.hashtag}: {trend.momentum_score:.1f}")

    # Recommendations
    use_now = [t for t in trending if t.recommendation == "use_now"]
    if use_now:
        print(f"\n⚡ USE NOW ({len(use_now)} hashtags):")
        for trend in use_now[:5]:
            print(f"  #{trend.hashtag} - {trend.stage.value}, momentum {trend.momentum_score:.1f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover trending hashtags"
    )
    parser.add_argument(
        "--platform",
        "-p",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform name",
    )
    parser.add_argument(
        "--category",
        "-c",
        help="Category filter (optional)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum results (default: 20)",
    )

    args = parser.parse_args()

    asyncio.run(
        discover_trending(
            platform=args.platform,
            category=args.category,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()