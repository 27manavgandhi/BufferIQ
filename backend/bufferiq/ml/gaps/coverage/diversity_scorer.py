"""Content diversity scoring."""

from typing import List
import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class DiversityScorer:
    """
    Calculate content diversity using Shannon entropy.

    Measures variety in content types, topics, and formats.
    """

    def calculate_diversity(self, items: List[str]) -> float:
        """
        Calculate Shannon entropy for diversity.

        Args:
            items: List of item labels

        Returns:
            Diversity score (0-1)
        """
        if not items:
            return 0.0

        # Count frequencies
        counts = Counter(items)
        total = len(items)

        # Calculate probabilities
        probabilities = [count / total for count in counts.values()]

        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0

        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        return round(diversity, 3)

    def calculate_category_diversity(
        self, categories: List[str], weights: List[float] = None
    ) -> float:
        """
        Calculate weighted category diversity.

        Args:
            categories: List of categories
            weights: Optional weights for each category

        Returns:
            Weighted diversity score
        """
        if not categories:
            return 0.0

        if weights is None:
            weights = [1.0] * len(categories)

        # Ensure weights match categories
        if len(weights) != len(categories):
            weights = [1.0] * len(categories)

        # Count weighted frequencies
        weighted_counts: dict = {}
        for cat, weight in zip(categories, weights):
            weighted_counts[cat] = weighted_counts.get(cat, 0.0) + weight

        total_weight = sum(weighted_counts.values())

        if total_weight == 0:
            return 0.0

        # Calculate probabilities
        probabilities = [count / total_weight for count in weighted_counts.values()]

        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Normalize
        max_entropy = (
            np.log2(len(weighted_counts)) if len(weighted_counts) > 1 else 1.0
        )

        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        return round(diversity, 3)

    def calculate_temporal_diversity(
        self, timestamps: List[str], window_days: int = 7
    ) -> float:
        """
        Calculate temporal diversity (how spread out content is over time).

        Args:
            timestamps: List of ISO timestamp strings
            window_days: Window size in days

        Returns:
            Temporal diversity score
        """
        # Simplified implementation
        # In production, would analyze actual temporal distribution

        if len(timestamps) < 2:
            return 0.0

        # For now, return moderate diversity
        return 0.65