"""
Content suggestion generation.

Generates actionable suggestions for content improvement.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class ContentSuggestion:
    """Content optimization suggestion."""

    type: str  # "sentiment", "length", "readability", "hashtags", "timing"
    priority: str  # "high", "medium", "low"
    current_value: Any
    suggested_value: Any
    impact: str  # Description of expected impact
    confidence: float


class SuggestionGenerator:
    """
        Generate content optimization suggestions.

        Analyzes content and provides actionable recommendations.

        Example:
    ```python
            generator = SuggestionGenerator()
            suggestions = generator.generate(
                text="Check this",
                platform="linkedin",
                analysis=analysis_result
            )
            for suggestion in suggestions:
                print(f"{suggestion.type}: {suggestion.impact}")
    ```
    """

    def __init__(self) -> None:
        """Initialize suggestion generator."""
        self.platform_best_practices = {
            "linkedin": {
                "ideal_length": (150, 250),
                "ideal_hashtags": (3, 5),
                "ideal_sentiment": "positive",
            },
            "twitter": {
                "ideal_length": (100, 200),
                "ideal_hashtags": (1, 3),
                "ideal_sentiment": "neutral",
            },
            "bluesky": {
                "ideal_length": (100, 200),
                "ideal_hashtags": (1, 3),
                "ideal_sentiment": "neutral",
            },
        }

    def generate(
        self, text: str, platform: str, analysis: Dict[str, Any]
    ) -> List[ContentSuggestion]:
        """
        Generate optimization suggestions.

        Args:
            text: Content text
            platform: Platform type
            analysis: Analysis results

        Returns:
            List of suggestions

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        suggestions = []
        best_practices = self.platform_best_practices[platform]

        # Length suggestions
        text_length = len(text)
        ideal_min, ideal_max = best_practices["ideal_length"]

        if text_length < ideal_min:
            suggestions.append(
                ContentSuggestion(
                    type="length",
                    priority="medium",
                    current_value=text_length,
                    suggested_value=f"{ideal_min}-{ideal_max} characters",
                    impact=f"Adding {ideal_min - text_length}+ characters may improve engagement",
                    confidence=0.7,
                )
            )
        elif text_length > ideal_max:
            suggestions.append(
                ContentSuggestion(
                    type="length",
                    priority="medium",
                    current_value=text_length,
                    suggested_value=f"{ideal_min}-{ideal_max} characters",
                    impact=f"Reducing by {text_length - ideal_max} characters may improve readability",
                    confidence=0.7,
                )
            )

        # Sentiment suggestions
        if "sentiment" in analysis:
            sentiment = analysis["sentiment"]["sentiment"]
            ideal_sentiment = best_practices["ideal_sentiment"]

            if sentiment != ideal_sentiment:
                suggestions.append(
                    ContentSuggestion(
                        type="sentiment",
                        priority="low",
                        current_value=sentiment,
                        suggested_value=ideal_sentiment,
                        impact=f"Adjusting tone to {ideal_sentiment} may align better with {platform}",
                        confidence=0.6,
                    )
                )

        # Hashtag suggestions
        if "features" in analysis:
            features = analysis["features"]
            hashtag_count = features.get("hashtag_count", 0)
            ideal_min_tags, ideal_max_tags = best_practices["ideal_hashtags"]

            if hashtag_count < ideal_min_tags:
                suggestions.append(
                    ContentSuggestion(
                        type="hashtags",
                        priority="medium",
                        current_value=hashtag_count,
                        suggested_value=f"{ideal_min_tags}-{ideal_max_tags} hashtags",
                        impact=f"Adding {ideal_min_tags - hashtag_count}+ hashtags may improve discoverability",
                        confidence=0.75,
                    )
                )
            elif hashtag_count > ideal_max_tags:
                suggestions.append(
                    ContentSuggestion(
                        type="hashtags",
                        priority="low",
                        current_value=hashtag_count,
                        suggested_value=f"{ideal_min_tags}-{ideal_max_tags} hashtags",
                        impact=f"Reducing to {ideal_max_tags} hashtags may look less spammy",
                        confidence=0.65,
                    )
                )

        # Readability suggestions
        if "readability" in analysis:
            readability = analysis["readability"]
            if readability.get("reading_difficulty") == "hard":
                suggestions.append(
                    ContentSuggestion(
                        type="readability",
                        priority="high",
                        current_value="hard",
                        suggested_value="medium or easy",
                        impact="Simplifying language may reach a wider audience",
                        confidence=0.8,
                    )
                )

        return suggestions
