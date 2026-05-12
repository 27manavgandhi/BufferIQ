"""
Topic diversity analysis.

Measures diversity of topics across content.
"""

from collections import Counter
from typing import Dict, List

import numpy as np


class TopicDiversityAnalyzer:
    """
        Analyze topic diversity.

        Uses Shannon entropy to measure topic diversity.

        Example:
    ```python
            analyzer = TopicDiversityAnalyzer()
            topics = ["AI", "AI", "ML", "Data", "AI"]
            diversity = analyzer.calculate_diversity(topics)
            print(f"Diversity: {diversity:.2f}")
    ```
    """

    def __init__(self) -> None:
        """Initialize topic diversity analyzer."""
        pass

    def calculate_diversity(self, topics: List[str]) -> float:
        """
        Calculate topic diversity using Shannon entropy.

        Args:
            topics: List of topic labels

        Returns:
            Diversity score (0-1), higher = more diverse

        Raises:
            ValueError: If topics list is empty
        """
        if not topics:
            raise ValueError("Topics list cannot be empty")

        # Count topic frequencies
        topic_counts = Counter(topics)
        total = len(topics)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in topic_counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability)

        # Normalize to 0-1 range
        max_entropy = np.log2(len(topic_counts)) if len(topic_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def calculate_topic_distribution(self, topics: List[str]) -> Dict[str, float]:
        """
        Calculate topic distribution.

        Args:
            topics: List of topic labels

        Returns:
            Dictionary of topic frequencies

        Raises:
            ValueError: If topics list is empty
        """
        if not topics:
            raise ValueError("Topics list cannot be empty")

        topic_counts = Counter(topics)
        total = len(topics)

        return {topic: count / total for topic, count in topic_counts.items()}

    def calculate_gini_coefficient(self, topics: List[str]) -> float:
        """
        Calculate Gini coefficient for topic inequality.

        Args:
            topics: List of topic labels

        Returns:
            Gini coefficient (0-1), 0 = perfectly equal

        Raises:
            ValueError: If topics list is empty
        """
        if not topics:
            raise ValueError("Topics list cannot be empty")

        # Get topic counts
        topic_counts = Counter(topics)
        counts = sorted(topic_counts.values())

        n = len(counts)
        if n == 0:
            return 0.0

        # Calculate Gini coefficient
        cumsum = np.cumsum(counts)
        gini = (2 * np.sum((np.arange(1, n + 1) * counts))) / (n * cumsum[-1]) - (
            n + 1
        ) / n

        return gini
