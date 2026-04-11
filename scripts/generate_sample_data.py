"""Generate realistic sample data for testing and development."""

import argparse
import asyncio
import hashlib
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from bufferiq.core.config import get_settings
from bufferiq.core.database import DatabaseManager
from bufferiq.domain.models import Channel, Organization, Post, User


async def generate_sample_data(
    num_posts: int = 500,
    platforms: list[str] | None = None,
    date_range_days: int = 180,
) -> None:
    """
    Generate realistic sample posts.

    Args:
        num_posts: Number of posts to generate
        platforms: List of platforms (default: linkedin, twitter, facebook)
        date_range_days: Days back to generate data for
    """
    if platforms is None:
        platforms = ["linkedin", "twitter", "facebook"]

    settings = get_settings()
    db_manager = DatabaseManager(settings)

    await db_manager.connect()

    try:
        async with db_manager.session() as session:
            # Create or get test user
            result = await session.execute(
                select(User).where(User.email == "sample@example.com")
            )
            user = result.scalar_one_or_none()

            if user is None:
                user = User(
                    buffer_org_id="sample_org",
                    buffer_access_token="sample_token",
                    email="sample@example.com",
                )
                session.add(user)
                await session.flush()

            # Create organization
            result = await session.execute(
                select(Organization).where(
                    Organization.buffer_org_id == "sample_org"
                )
            )
            org = result.scalar_one_or_none()

            if org is None:
                org = Organization(
                    user_id=user.id,
                    buffer_org_id="sample_org",
                    name="Sample Organization",
                )
                session.add(org)
                await session.flush()

            # Create channels for each platform
            channels = {}
            for platform in platforms:
                result = await session.execute(
                    select(Channel).where(
                        Channel.buffer_channel_id == f"sample_{platform}"
                    )
                )
                channel = result.scalar_one_or_none()

                if channel is None:
                    channel = Channel(
                        organization_id=org.id,
                        buffer_channel_id=f"sample_{platform}",
                        platform=platform,
                        handle=f"@sample_{platform}",
                    )
                    session.add(channel)
                    await session.flush()

                channels[platform] = channel

            # Generate posts
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=date_range_days)

            # Platform-specific characteristics
            platform_stats = {
                "linkedin": {
                    "base_engagement": 0.04,
                    "variance": 0.02,
                    "avg_length": 800,
                    "best_hours": [8, 9, 10, 11, 12, 17],
                    "best_days": [1, 2, 3, 4],  # Tuesday-Friday
                },
                "twitter": {
                    "base_engagement": 0.02,
                    "variance": 0.015,
                    "avg_length": 200,
                    "best_hours": [9, 12, 15, 18, 20],
                    "best_days": [0, 1, 2, 3, 4],  # Monday-Friday
                },
                "facebook": {
                    "base_engagement": 0.03,
                    "variance": 0.018,
                    "avg_length": 400,
                    "best_hours": [13, 15, 19, 20],
                    "best_days": [2, 3, 4, 5, 6],  # Wednesday-Sunday
                },
            }

            content_templates = [
                "Excited to share {topic}! {hashtags}",
                "Just launched {topic}. Check it out! {url}",
                "Thoughts on {topic}? {question}",
                "Here's what we learned from {topic}: {insights}",
                "{topic} - a thread 🧵 {hashtags}",
                "New blog post: {topic} {url} {hashtags}",
                "Quick tip: {tip} {hashtags}",
                "{topic} - what do you think? {question}",
            ]

            topics = [
                "AI and Machine Learning",
                "Social Media Strategy",
                "Content Marketing",
                "Developer Tools",
                "Productivity Hacks",
                "Team Collaboration",
                "Remote Work",
                "Digital Transformation",
                "Customer Success",
                "Product Updates",
            ]

            hashtags_pool = [
                "#AI",
                "#MachineLearning",
                "#SocialMedia",
                "#Marketing",
                "#DevTools",
                "#Productivity",
                "#RemoteWork",
                "#Tech",
                "#Innovation",
                "#Growth",
            ]

            print(f"Generating {num_posts} sample posts...")

            for i in range(num_posts):
                # Random platform
                platform = random.choice(platforms)
                channel = channels[platform]
                stats = platform_stats[platform]

                # Random date within range
                days_back = random.randint(0, date_range_days)
                published_at = (end_date - timedelta(days=days_back)).replace(tzinfo=None)

                # Adjust hour based on platform best times
                if random.random() < 0.6:  # 60% chance of optimal time
                    hour = random.choice(stats["best_hours"])
                else:
                    hour = random.randint(0, 23)

                published_at = published_at.replace(hour=hour, minute=random.randint(0, 59))

                # Adjust day based on platform best days
                if published_at.weekday() in stats["best_days"]:
                    day_boost = 1.2
                else:
                    day_boost = 0.8

                if hour in stats["best_hours"]:
                    hour_boost = 1.3
                else:
                    hour_boost = 0.9

                # Generate content
                template = random.choice(content_templates)
                topic = random.choice(topics)

                num_hashtags = random.randint(0, 4)
                hashtags = " ".join(random.sample(hashtags_pool, min(num_hashtags, len(hashtags_pool))))

                url = "https://example.com/article" if random.random() < 0.4 else ""
                question = "What's your experience?" if "?" in template else ""
                tip = "Always test before deploying" if "tip" in template else ""
                insights = "Consistency is key" if "learned" in template else ""

                content = template.format(
                    topic=topic,
                    hashtags=hashtags,
                    url=url,
                    question=question,
                    tip=tip,
                    insights=insights,
                )

                # Trim to platform length
                if len(content) > stats["avg_length"] * 1.5:
                    content = content[: int(stats["avg_length"] * 1.5)]

                # Generate engagement metrics
                base_engagement = stats["base_engagement"] * day_boost * hour_boost

                # Add length bonus
                if len(content) > stats["avg_length"] * 0.8:
                    length_boost = 1.1
                else:
                    length_boost = 0.95

                # Add hashtag bonus
                if num_hashtags > 0 and num_hashtags <= 3:
                    hashtag_boost = 1.15
                else:
                    hashtag_boost = 1.0

                final_engagement = (
                    base_engagement
                    * length_boost
                    * hashtag_boost
                    * random.uniform(0.7, 1.3)
                )

                # Generate impressions (log-normal distribution)
                impressions = int(random.lognormvariate(6, 1.5))  # Mean ~1000

                # Calculate engagement numbers
                total_engagement = int(impressions * final_engagement)
                likes = int(total_engagement * random.uniform(0.6, 0.8))
                comments = int(total_engagement * random.uniform(0.1, 0.2))
                shares = total_engagement - likes - comments
                clicks = int(impressions * random.uniform(0.01, 0.05))

                engagement_rate = (
                    total_engagement / impressions if impressions > 0 else 0.0
                )

                # Create post
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                post = Post(
                    channel_id=channel.id,
                    buffer_post_id=f"sample_post_{i}_{platform}",
                    content=content,
                    content_hash=content_hash,
                    status="sent",
                    published_at=published_at,
                    likes=likes,
                    comments=comments,
                    shares=shares,
                    clicks=clicks,
                    impressions=impressions,
                    engagement_rate=engagement_rate,
                )

                session.add(post)

                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_posts} posts...")
                    await session.flush()

            await session.commit()
            print(f"✅ Successfully generated {num_posts} sample posts!")

    finally:
        await db_manager.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate sample data for BufferIQ")
    parser.add_argument(
        "--posts", type=int, default=500, help="Number of posts to generate"
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["linkedin", "twitter", "facebook"],
        help="Platforms to generate data for",
    )
    parser.add_argument(
        "--days", type=int, default=180, help="Date range in days"
    )

    args = parser.parse_args()

    asyncio.run(
        generate_sample_data(
            num_posts=args.posts, platforms=args.platforms, date_range_days=args.days
        )
    )


if __name__ == "__main__":
    main()
