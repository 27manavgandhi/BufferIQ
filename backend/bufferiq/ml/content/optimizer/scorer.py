"""
Content scoring.

Scores content based on multiple factors.
"""

from typing import Any, Dict

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ContentScorer:
    """
        Score content quality.

        Provides an overall quality score based on multiple factors.

        Example:
    ```python
            scorer = ContentScorer()
            score = scorer.score(
                text="Great post!",
                platform="linkedin",
                analysis=analysis_result
            )
            print(f"Score: {score}/100")
    ```
    """

    def __init__(self) -> None:
        """Initialize content scorer."""
        # Weights for different factors
        self.weights = {
            "quality": 0.3,
            "readability": 0.2,
            "sentiment": 0.15,
            "length": 0.15,
            "features": 0.2,
        }

    def score(self, text: str, platform: str, analysis: Dict[str, Any]) -> float:
        """
        Score content.

        Args:
            text: Content text
            platform: Platform type
            analysis: Analysis results

        Returns:
            Score (0-100)

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        scores = {}

        # Quality score
        if "quality" in analysis:
            scores["quality"] = analysis["quality"].get("score", 50.0)
        else:
            scores["quality"] = 50.0

        # Readability score
        if "readability" in analysis:
            readability = analysis["readability"]
            # Convert difficulty to score
            if readability.get("reading_difficulty") == "easy":
                scores["readability"] = 90.0
            elif readability.get("reading_difficulty") == "medium":
                scores["readability"] = 75.0
            else:
                scores["readability"] = 50.0
        else:
            scores["readability"] = 50.0

        # Sentiment score
        if "sentiment" in analysis:
            confidence = analysis["sentiment"].get("confidence", 0.5)
            scores["sentiment"] = confidence * 100
        else:
            scores["sentiment"] = 50.0

        # Length score
        text_length = len(text)
        scores["length"] = self._score_length(text_length, platform)

        # Features score
        if "features" in analysis:
            features = analysis["features"]
            scores["features"] = self._score_features(features, platform)
        else:
            scores["features"] = 50.0

        # Calculate weighted average
        overall_score = sum(
            scores[factor] * self.weights[factor] for factor in self.weights
        )

        return round(overall_score, 1)

    def _score_length(self, length: int, platform: str) -> float:
        """Score based on text length."""
        ideal_ranges = {
            "linkedin": (150, 250),
            "twitter": (100, 200),
            "bluesky": (100, 200),
        }

        ideal_min, ideal_max = ideal_ranges[platform]

        if ideal_min <= length <= ideal_max:
            return 100.0
        elif length < ideal_min:
            return (length / ideal_min) * 100
        else:
            penalty = (length - ideal_max) / ideal_max
            return max(0.0, 100.0 - (penalty * 50))

    def _score_features(self, features: Dict[str, Any], platform: str) -> float:
        """Score based on content features."""
        score = 50.0

        # Bonus for hashtags
        hashtag_count = features.get("hashtag_count", 0)
        if 1 <= hashtag_count <= 5:
            score += 20

        # Bonus for emojis (moderate use)
        emoji_count = features.get("emoji_count", 0)
        if 1 <= emoji_count <= 3:
            score += 15

        # Bonus for URLs (if not excessive)
        url_count = features.get("url_count", 0)
        if url_count == 1:
            score += 15

        return min(100.0, score)
