"""CLI commands for feature engineering."""

import asyncio
import sys
from pathlib import Path

import click
import pandas as pd
from sqlalchemy import select

from bufferiq.core.database import async_session_maker
from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Post
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline
from bufferiq.ml.features.scaler import FeatureScaler
from bufferiq.ml.features.selector import FeatureSelector

logger = get_logger(__name__)


@click.group()
def features() -> None:
    """Feature engineering commands."""
    pass


@features.command()
@click.option(
    "--user-id", type=int, required=True, help="User ID to extract features for"
)
@click.option("--output", type=str, help="Output CSV file path")
@click.option(
    "--extractors",
    type=str,
    help="Comma-separated list of extractors (temporal,content,nlp,engagement,platform)",
)
@click.option("--stats", is_flag=True, help="Show feature statistics")
@click.option("--fit-scaler", is_flag=True, help="Fit scaler on this data")
@click.option("--save-scaler", type=str, help="Path to save fitted scaler")
@click.option("--fit-selector", is_flag=True, help="Fit feature selector")
@click.option("--save-selector", type=str, help="Path to save fitted selector")
@click.option(
    "--target", type=str, default="engagement_rate", help="Target column for selector"
)
def extract(
    user_id: int,
    output: str | None,
    extractors: str | None,
    stats: bool,
    fit_scaler: bool,
    save_scaler: str | None,
    fit_selector: bool,
    save_selector: str | None,
    target: str,
) -> None:
    """Extract features for user posts."""
    asyncio.run(
        _extract_features(
            user_id=user_id,
            output=output,
            extractors=extractors,
            stats=stats,
            fit_scaler=fit_scaler,
            save_scaler=save_scaler,
            fit_selector=fit_selector,
            save_selector=save_selector,
            target=target,
        )
    )


async def _extract_features(
    user_id: int,
    output: str | None,
    extractors: str | None,
    stats: bool,
    fit_scaler: bool,
    save_scaler: str | None,
    fit_selector: bool,
    save_selector: str | None,
    target: str,
) -> None:
    """Async feature extraction."""
    logger.info(f"Extracting features for user {user_id}")

    # Fetch posts
    async with async_session_maker() as session:
        stmt = select(Post).where(Post.user_id == user_id, Post.status == "sent")
        result = await session.execute(stmt)
        posts = result.scalars().all()

    if not posts:
        logger.error(f"No posts found for user {user_id}")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "published_at": post.sent_at or post.scheduled_at,
                "content": post.content,
                "platform": post.platform,
                "likes": post.likes,
                "comments": post.comments,
                "shares": post.shares,
                "impressions": post.impressions,
                "clicks": post.clicks,
            }
            for post in posts
        ]
    )

    # Calculate engagement rate
    df["engagement_rate"] = (
        (df["likes"] + df["comments"] + df["shares"])
        / df["impressions"].replace(0, 1)
        * 100
    )

    logger.info(f"Loaded {len(df)} posts")

    # Create pipeline
    scaler_obj = FeatureScaler(method="standard") if fit_scaler or save_scaler else None
    selector_obj = (
        FeatureSelector(method="mutual_info", k=20)
        if fit_selector or save_selector
        else None
    )

    # Filter extractors if specified
    if extractors:
        from bufferiq.ml.features.content import ContentFeatureExtractor
        from bufferiq.ml.features.engagement import EngagementFeatureExtractor
        from bufferiq.ml.features.nlp import NLPFeatureExtractor
        from bufferiq.ml.features.platform_specific import (
            PlatformSpecificFeatureExtractor,
        )
        from bufferiq.ml.features.temporal import TemporalFeatureExtractor

        extractor_map = {
            "temporal": TemporalFeatureExtractor,
            "content": ContentFeatureExtractor,
            "nlp": NLPFeatureExtractor,
            "engagement": EngagementFeatureExtractor,
            "platform": PlatformSpecificFeatureExtractor,
        }

        extractor_list = [
            extractor_map[name.strip()]()
            for name in extractors.split(",")
            if name.strip() in extractor_map
        ]

        pipeline = FeatureEngineeringPipeline(
            extractors=extractor_list, scaler=scaler_obj, selector=selector_obj
        )
    else:
        pipeline = FeatureEngineeringPipeline(scaler=scaler_obj, selector=selector_obj)

    # Extract features
    async with async_session_maker() as session:
        features = await pipeline.extract_features(
            df,
            session=session,
            fit_scaler=fit_scaler or save_scaler is not None,
            fit_selector=fit_selector or save_selector is not None,
            target_column=target if (fit_selector or save_selector) else None,
        )

    logger.info(f"Extracted {len(features.columns)} features")

    # Show statistics
    if stats:
        click.echo("\nFeature Statistics:")
        click.echo(f"Total features: {len(features.columns)}")
        click.echo(f"Total rows: {len(features)}")
        click.echo("\nFeature breakdown:")
        stats_dict = pipeline.get_feature_stats()
        for key, value in stats_dict.items():
            click.echo(f"  {key}: {value}")

    # Save scaler
    if save_scaler and pipeline.scaler:
        pipeline.scaler.save(save_scaler)
        logger.info(f"Saved scaler to {save_scaler}")

    # Save selector
    if save_selector and pipeline.selector:
        pipeline.selector.save(save_selector)
        logger.info(f"Saved selector to {save_selector}")

    # Save output
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False)
        logger.info(f"Saved features to {output}")
        click.echo(f"\n✅ Features saved to {output}")
    else:
        click.echo("\nFeature Preview:")
        click.echo(features.head().to_string())


@features.command()
def list_features() -> None:
    """List all available features."""
    pipeline = FeatureEngineeringPipeline()
    all_features = pipeline.get_all_feature_names()

    click.echo(f"\nTotal features: {len(all_features)}\n")

    # Group by extractor
    from bufferiq.ml.features.content import ContentFeatureExtractor
    from bufferiq.ml.features.engagement import EngagementFeatureExtractor
    from bufferiq.ml.features.nlp import NLPFeatureExtractor
    from bufferiq.ml.features.platform_specific import PlatformSpecificFeatureExtractor
    from bufferiq.ml.features.temporal import TemporalFeatureExtractor

    extractors = [
        ("Temporal", TemporalFeatureExtractor()),
        ("Content", ContentFeatureExtractor()),
        ("NLP", NLPFeatureExtractor()),
        ("Engagement", EngagementFeatureExtractor()),
        ("Platform-Specific", PlatformSpecificFeatureExtractor()),
    ]

    for name, extractor in extractors:
        features = extractor.feature_names
        click.echo(f"{name} Features ({len(features)}):")
        for i, feature in enumerate(features, 1):
            click.echo(f"  {i}. {feature}")
        click.echo()


@features.command()
@click.option("--user-id", type=int, required=True, help="User ID")
@click.option("--target", type=str, default="engagement_rate", help="Target column")
@click.option("--top", type=int, default=20, help="Number of top features to show")
@click.option("--output", type=str, help="Output CSV file path")
def importance(user_id: int, target: str, top: int, output: str | None) -> None:
    """Analyze feature importance."""
    asyncio.run(_analyze_importance(user_id, target, top, output))


async def _analyze_importance(
    user_id: int, target: str, top: int, output: str | None
) -> None:
    """Async importance analysis."""
    # First extract features
    async with async_session_maker() as session:
        stmt = select(Post).where(Post.user_id == user_id, Post.status == "sent")
        result = await session.execute(stmt)
        posts = result.scalars().all()

    if not posts:
        logger.error(f"No posts found for user {user_id}")
        sys.exit(1)

    df = pd.DataFrame(
        [
            {
                "published_at": post.sent_at or post.scheduled_at,
                "content": post.content,
                "platform": post.platform,
                "likes": post.likes,
                "comments": post.comments,
                "shares": post.shares,
                "impressions": post.impressions,
                "clicks": post.clicks,
            }
            for post in posts
        ]
    )

    df["engagement_rate"] = (
        (df["likes"] + df["comments"] + df["shares"])
        / df["impressions"].replace(0, 1)
        * 100
    )

    # Extract features
    pipeline = FeatureEngineeringPipeline()
    async with async_session_maker() as session:
        features = await pipeline.extract_features(df, session=session)

    # Fit selector to get importance
    selector = FeatureSelector(method="mutual_info", k=top)
    selector.fit(features, df[target])

    importance_df = selector.get_feature_importance()
    top_features = importance_df.head(top)

    click.echo(f"\nTop {top} Features for {target}:\n")
    for i, row in enumerate(top_features.itertuples(), 1):
        click.echo(f"{i}. {row.feature}: {row.importance:.4f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output, index=False)
        logger.info(f"Saved feature importance to {output}")


if __name__ == "__main__":
    features()
