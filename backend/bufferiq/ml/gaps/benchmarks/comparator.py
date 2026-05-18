"""Benchmark comparison."""

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class BenchmarkComparator:
    """
    Compare performance against benchmarks.

    Compares user metrics against industry standards and competitors.
    """

    def compare_to_industry(
        self,
        user_metrics: Dict[str, float],
        industry: str = "technology",
    ) -> Dict[str, Any]:
        """
        Compare to industry benchmarks.

        Args:
            user_metrics: User's metrics
            industry: Industry category

        Returns:
            Comparison results
        """
        # Industry benchmarks (mock data)
        industry_benchmarks = self._get_industry_benchmarks(industry)

        comparisons = {}

        for metric, user_value in user_metrics.items():
            if metric in industry_benchmarks:
                industry_value = industry_benchmarks[metric]

                # Calculate variance
                variance = ((user_value - industry_value) / industry_value * 100) if industry_value > 0 else 0

                comparisons[metric] = {
                    "user_value": user_value,
                    "industry_average": industry_value,
                    "variance_percent": round(variance, 2),
                    "position": "above" if variance > 0 else "below" if variance < 0 else "at",
                }

        # Overall position
        above_count = sum(1 for c in comparisons.values() if c["position"] == "above")
        total_count = len(comparisons)

        if above_count / total_count >= 0.6:
            overall_position = "above_average"
        elif above_count / total_count <= 0.4:
            overall_position = "below_average"
        else:
            overall_position = "average"

        return {
            "industry": industry,
            "comparisons": comparisons,
            "overall_position": overall_position,
        }

    def _get_industry_benchmarks(self, industry: str) -> Dict[str, float]:
        """Get industry benchmark values."""
        benchmarks = {
            "technology": {
                "engagement_rate": 0.048,
                "posts_per_week": 3.5,
                "avg_likes": 90.0,
                "avg_comments": 15.0,
            },
            "marketing": {
                "engagement_rate": 0.055,
                "posts_per_week": 4.2,
                "avg_likes": 110.0,
                "avg_comments": 18.0,
            },
        }

        return benchmarks.get(industry, benchmarks["technology"])

    def compare_to_competitors(
        self,
        user_metrics: Dict[str, float],
        competitor_metrics: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compare to specific competitors.

        Args:
            user_metrics: User's metrics
            competitor_metrics: List of competitor metric dicts

        Returns:
            Competitive comparison
        """
        if not competitor_metrics:
            return {"error": "No competitor data available"}

        # Calculate competitor averages
        avg_metrics = {}
        for metric in user_metrics.keys():
            values = [c.get(metric, 0) for c in competitor_metrics if metric in c]
            if values:
                avg_metrics[metric] = sum(values) / len(values)

        # Compare
        comparisons = {}
        for metric, user_value in user_metrics.items():
            if metric in avg_metrics:
                comp_avg = avg_metrics[metric]
                variance = ((user_value - comp_avg) / comp_avg * 100) if comp_avg > 0 else 0

                comparisons[metric] = {
                    "user_value": user_value,
                    "competitor_average": comp_avg,
                    "variance_percent": round(variance, 2),
                    "rank": self._calculate_rank(user_value, competitor_metrics, metric),
                }

        return {
            "comparisons": comparisons,
            "competitor_count": len(competitor_metrics),
        }

    def _calculate_rank(
        self,
        user_value: float,
        competitor_metrics: List[Dict[str, float]],
        metric: str,
    ) -> int:
        """Calculate user's rank for a metric."""
        all_values = [c.get(metric, 0) for c in competitor_metrics if metric in c]
        all_values.append(user_value)
        all_values.sort(reverse=True)

        return all_values.index(user_value) + 1