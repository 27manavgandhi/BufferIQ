"""
Hashtag rotation scheduler.

Generates rotation schedules to keep content fresh.
"""

from typing import Dict, List
from datetime import datetime, timedelta


class HashtagRotator:
    """
    Generate hashtag rotation schedules.

    Rotates hashtags to avoid repetition and maintain freshness.

    Example:
```python
        rotator = HashtagRotator()
        schedule = rotator.create_schedule(
            hashtag_pool=["ai", "tech", "ml", "data", "innovation"],
            posts_per_week=3,
            weeks=4
        )

        for week, hashtags in schedule.items():
            print(f"Week {week}: {hashtags}")
```
    """

    def create_schedule(
        self,
        hashtag_pool: List[str],
        posts_per_week: int,
        weeks: int,
        hashtags_per_post: int = 5,
    ) -> Dict[str, List[List[str]]]:
        """
        Create rotation schedule.

        Args:
            hashtag_pool: Pool of hashtags to rotate
            posts_per_week: Number of posts per week
            weeks: Number of weeks
            hashtags_per_post: Hashtags per post

        Returns:
            Schedule as {week: [[post1_hashtags], [post2_hashtags], ...]}
        """
        schedule: Dict[str, List[List[str]]] = {}

        total_posts = posts_per_week * weeks
        hashtag_index = 0

        for week in range(1, weeks + 1):
            week_key = f"week_{week}"
            week_posts: List[List[str]] = []

            for _ in range(posts_per_week):
                # Select hashtags for this post
                post_hashtags: List[str] = []

                for _ in range(hashtags_per_post):
                    if hashtag_pool:
                        # Rotate through pool
                        hashtag = hashtag_pool[hashtag_index % len(hashtag_pool)]
                        post_hashtags.append(hashtag)
                        hashtag_index += 1

                week_posts.append(post_hashtags)

            schedule[week_key] = week_posts

        return schedule

    def optimize_rotation(
        self, hashtags: List[str], target_variety: float = 0.7
    ) -> List[str]:
        """
        Optimize hashtag rotation order.

        Args:
            hashtags: List of hashtags
            target_variety: Target variety score (0-1)

        Returns:
            Optimized order
        """
        # Simple optimization: shuffle to maximize spacing
        # In production, use more sophisticated algorithm

        # For now, just return original order
        # Real implementation would optimize spacing
        return hashtags.copy()