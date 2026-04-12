"""Script to extract features from posts."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
from sqlalchemy import select

from bufferiq.core.database import async_session_maker
from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Post
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline
from bufferiq.ml.features.scaler import FeatureScaler

logger = get_logger(__name__)


async def extract_features(
    user_id: int,
    platform: str | None = None,
    output: str | None = None,
    save_scaler: bool = False,
) -> None:
    """
    Extract features from user posts.

    Args:
        user_id: User ID to extract features for
        platform: Optional platform filter
        output: Optional output file path
        save_scaler: Whether to save fitted scaler
    """
    logger.info(f"Extracting features for user {user_id}")

    # Fetch posts
    async with async_session_maker() as session:
        stmt = select(Post).where(Post.user_id == user_id, Post.status == "sent")

        if platform:
            stmt = stmt.where(Post.platform == platform)

        result = await session.execute(stmt)
        posts = result.scalars().all()

    if not posts:
        logger.error(f"No posts found for user {user_id}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "id": post.id,
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

    logger.info(f"Loaded {len(df)} posts")

    # Create pipeline
    scaler = FeatureScaler(method="standard") if save_scaler else None
    pipeline = FeatureEngineeringPipeline(scaler=scaler)

    # Extract features
    async with async_session_maker() as session:
        features = await pipeline.extract_features(
            df, session=session, fit_scaler=save_scaler
        )

    logger.info(f"Extracted {len(features.columns)} features")

    # Add post ID for reference
    features.insert(0, "post_id", df["id"])

    # Print statistics
    print("\n" + "=" * 60)
    print("Feature Extraction Summary")
    print("=" * 60)
    print(f"Total posts: {len(features)}")
    print(f"Total features: {len(features.columns) - 1}")  # Exclude post_id
    print(f"\nFeature breakdown:")
    stats = pipeline.get_feature_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save output
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        print(f"\n✅ Features saved to {output_path}")

        # Save scaler if requested
        if save_scaler and pipeline.scaler:
            scaler_path = output_path.parent / "scaler.joblib"
            pipeline.scaler.save(str(scaler_path))
            logger.info(f"Saved scaler to {scaler_path}")
            print(f"✅ Scaler saved to {scaler_path}")

    else:
        print("\nFeature preview:")
        print(features.head().to_string())
        print(f"\n... {len(features) - 5} more rows")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract features from posts")
    parser.add_argument("--user-id", type=int, required=True, help="User ID")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform filter",
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument(
        "--save-scaler", action="store_true", help="Save fitted scaler"
    )

    args = parser.parse_args()

    asyncio.run(
        extract_features(
            user_id=args.user_id,
            platform=args.platform,
            output=args.output,
            save_scaler=args.save_scaler,
        )
    )


if __name__ == "__main__":
    main()
