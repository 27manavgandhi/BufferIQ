"""
Readability scoring.

Provides simplified readability scores and recommendations.
"""

from typing import Dict

from bufferiq.ml.content.readability.analyzer import (
    ReadabilityAnalyzer,
    ReadabilityScores,
)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ReadabilityScorer:
    """
        Score and recommend readability improvements.

        Provides platform-specific readability recommendations.

        Example:
    ```python
            scorer = ReadabilityScorer()
            result = scorer.score(text, platform="linkedin")
            print(result["score"])  # 75.0
            print(result["recommendations"])
    ```
    """

    def __init__(self) -> None:
        """Initialize readability scorer."""
        self.analyzer = ReadabilityAnalyzer()

        # Platform-specific targets
        self.platform_targets = {
            "linkedin": {"max_grade": 12, "target_ease": 60},
            "twitter": {"max_grade": 8, "target_ease": 70},
            "bluesky": {"max_grade": 8, "target_ease": 70},
        }

    def score(self, text: str, platform: str) -> Dict[str, any]:
        """
        Score readability for platform.

        Args:
            text: Text to score
            platform: Platform type

        Returns:
            Score and recommendations

        Raises:
            ValueError: If platform not supported or text invalid
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        if not text or len(text.strip()) < 10:
            raise ValueError("Text is too short for scoring")

        # Analyze readability
        scores = self.analyzer.analyze(text)

        # Get platform targets
        targets = self.platform_targets[platform]

        # Calculate score (0-100)
        grade_score = self._score_grade_level(
            scores.average_grade_level, targets["max_grade"]
        )
        ease_score = self._score_ease(
            scores.flesch_reading_ease, targets["target_ease"]
        )

        overall_score = (grade_score + ease_score) / 2

        # Generate recommendations
        recommendations = self._generate_recommendations(scores, targets, platform)

        return {
            "score": overall_score,
            "scores": scores,
            "recommendations": recommendations,
            "platform_targets": targets,
        }

    def _score_grade_level(self, grade: float, max_grade: float) -> float:
        """Score based on grade level."""
        if grade <= max_grade:
            return 100.0
        else:
            # Penalize for being too complex
            penalty = (grade - max_grade) * 5
            return max(0.0, 100.0 - penalty)

    def _score_ease(self, ease: float, target: float) -> float:
        """Score based on reading ease."""
        if ease >= target:
            return 100.0
        else:
            # Scale based on target
            return (ease / target) * 100

    def _generate_recommendations(
        self, scores: ReadabilityScores, targets: Dict, platform: str
    ) -> list:
        """Generate readability recommendations."""
        recommendations = []

        if scores.average_grade_level > targets["max_grade"]:
            recommendations.append(
                f"Simplify language - current grade level "
                f"{scores.average_grade_level:.1f} exceeds "
                f"{platform} target of {targets['max_grade']}"
            )

        if scores.flesch_reading_ease < targets["target_ease"]:
            recommendations.append(
                f"Improve readability - current ease score "
                f"{scores.flesch_reading_ease:.1f} is below "
                f"target of {targets['target_ease']}"
            )

        if scores.reading_difficulty == "hard":
            recommendations.append("Use shorter sentences and simpler words")

        if not recommendations:
            recommendations.append("Readability is good for this platform")

        return recommendations
