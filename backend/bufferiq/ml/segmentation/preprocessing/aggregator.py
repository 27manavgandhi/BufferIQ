"""Engagement signal aggregation."""

from typing import Any, Dict, List

import numpy as np


class EngagementAggregator:
    """Aggregate engagement signals from multiple sources."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        """
        Initialize aggregator.

        Args:
            weights: Weights for different interaction types
        """
        self.weights = weights or {
            "likes": 0.3,
            "comments": 0.5,
            "shares": 0.8,
            "clicks": 0.4,
            "retweets": 0.6,
        }

    def aggregate(self, interaction_types: Dict[str, int]) -> float:
        """
        Aggregate engagement from interaction types.

        Args:
            interaction_types: Dict of interaction type counts

        Returns:
            Aggregated engagement score (0-1)
        """
        if not interaction_types:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for interaction_type, count in interaction_types.items():
            weight = self.weights.get(interaction_type, 0.5)
            weighted_sum += count * weight
            weight_total += weight

        if weight_total == 0:
            return 0.0

        # Normalize to 0-1 range
        return min(weighted_sum / weight_total / 10.0, 1.0)

    def normalize_engagement_rate(self, engagement_rate: float) -> float:
        """
        Normalize engagement rate to 0-1.

        Args:
            engagement_rate: Raw engagement rate

        Returns:
            Normalized rate (0-1)
        """
        return min(max(engagement_rate, 0.0), 1.0)