"""Topic saturation analysis."""

from typing import Any, Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SaturationAnalyzer:
    """
    Analyze topic saturation levels.

    Calculates how saturated each topic is based on posting frequency
    and temporal distribution.
    """

    def __init__(self, saturation_threshold: int = 20):
        """
        Initialize saturation analyzer.

        Args:
            saturation_threshold: Post count threshold for saturation
        """
        self.saturation_threshold = saturation_threshold

    def calculate_saturation(self, topics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate saturation score for each topic.

        Args:
            topics: List of topic dictionaries

        Returns:
            Dictionary mapping topic names to saturation scores (0-1)
        """
        saturation_scores = {}

        for topic in topics:
            topic_name = topic.get("name", "Unknown")
            post_count = topic.get("post_count", 0)

            # Calculate saturation based on post count
            saturation = min(post_count / self.saturation_threshold, 1.0)

            saturation_scores[topic_name] = round(saturation, 3)

        return saturation_scores

    def identify_oversaturated(
        self, saturation_scores: Dict[str, float], threshold: float = 0.8
    ) -> List[str]:
        """
        Identify oversaturated topics.

        Args:
            saturation_scores: Saturation scores by topic
            threshold: Saturation threshold

        Returns:
            List of oversaturated topic names
        """
        oversaturated = [
            topic for topic, score in saturation_scores.items() if score >= threshold
        ]

        return oversaturated

    def identify_undersaturated(
        self, saturation_scores: Dict[str, float], threshold: float = 0.3
    ) -> List[str]:
        """
        Identify undersaturated topics.

        Args:
            saturation_scores: Saturation scores by topic
            threshold: Saturation threshold

        Returns:
            List of undersaturated topic names
        """
        undersaturated = [
            topic for topic, score in saturation_scores.items() if score < threshold
        ]

        return undersaturated

    def calculate_balance_score(self, saturation_scores: Dict[str, float]) -> float:
        """
        Calculate overall topic balance score.

        Args:
            saturation_scores: Saturation scores

        Returns:
            Balance score (0-1), higher is more balanced
        """
        if not saturation_scores:
            return 0.0

        scores = list(saturation_scores.values())

        # Calculate coefficient of variation
        mean = np.mean(scores)
        std = np.std(scores)

        if mean == 0:
            return 0.0

        cv = std / mean

        # Convert to balance score (lower CV = higher balance)
        balance = 1 / (1 + cv)

        return round(balance, 3)