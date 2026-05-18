"""Competitor benchmarking."""

from typing import Any, Dict, List
import logging

from bufferiq.ml.gaps.competitors.analyzer import CompetitorProfile

logger = logging.getLogger(__name__)


class CompetitorBenchmarker:
    """
    Benchmark user performance against competitors.

    Compares key metrics and identifies strengths/weaknesses.
    """

    def benchmark(
        self, user_profile: CompetitorProfile, competitor_profiles: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """
        Benchmark user against competitors.

        Args:
            user_profile: User's profile
            competitor_profiles: Competitor profiles

        Returns:
            Benchmark results
        """
        if not competitor_profiles:
            return {
                "relative_performance": "no_comparison",
                "strengths": [],
                "weaknesses": [],
            }

        # Calculate averages
        avg_posts = sum(c.total_posts for c in competitor_profiles) / len(
            competitor_profiles
        )
        avg_engagement = sum(c.avg_engagement_rate for c in competitor_profiles) / len(
            competitor_profiles
        )
        avg_posts_per_week = sum(c.posts_per_week for c in competitor_profiles) / len(
            competitor_profiles
        )

        # Compare user
        strengths = []
        weaknesses = []

        if user_profile.total_posts > avg_posts:
            strengths.append(
                f"Higher total content volume ({user_profile.total_posts} vs {avg_posts:.0f} avg)"
            )
        else:
            weaknesses.append(
                f"Lower content volume ({user_profile.total_posts} vs {avg_posts:.0f} avg)"
            )

        if user_profile.avg_engagement_rate > avg_engagement:
            strengths.append(
                f"Higher engagement rate ({user_profile.avg_engagement_rate:.2%} vs {avg_engagement:.2%} avg)"
            )
        else:
            weaknesses.append(
                f"Lower engagement rate ({user_profile.avg_engagement_rate:.2%} vs {avg_engagement:.2%} avg)"
            )

        if user_profile.posts_per_week > avg_posts_per_week:
            strengths.append(
                f"Higher posting frequency ({user_profile.posts_per_week:.1f} vs {avg_posts_per_week:.1f} avg)"
            )
        else:
            weaknesses.append(
                f"Lower posting frequency ({user_profile.posts_per_week:.1f} vs {avg_posts_per_week:.1f} avg)"
            )

        # Determine relative performance
        strength_count = len(strengths)
        weakness_count = len(weaknesses)

        if strength_count > weakness_count:
            relative_performance = "above_average"
        elif weakness_count > strength_count:
            relative_performance = "below_average"
        else:
            relative_performance = "average"

        return {
            "relative_performance": relative_performance,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "avg_competitor_posts": round(avg_posts, 1),
            "avg_competitor_engagement": round(avg_engagement, 4),
            "avg_competitor_frequency": round(avg_posts_per_week, 1),
        }