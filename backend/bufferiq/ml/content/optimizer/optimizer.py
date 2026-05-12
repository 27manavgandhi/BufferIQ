"""
Content optimizer.

Main optimizer that coordinates all optimization components.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from bufferiq.ml.content.optimizer.suggestion_generator import (
    SuggestionGenerator,
    ContentSuggestion,
)
from bufferiq.ml.content.optimizer.scorer import ContentScorer

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class OptimizationResult:
    """Content optimization analysis."""

    overall_score: float  # 0-100
    suggestions: List[ContentSuggestion]
    predicted_engagement_lift: float  # % improvement
    best_platform: str
    best_time: Optional[datetime]
    rewrite_examples: List[str]


class ContentOptimizer:
    """
        Generate content optimization suggestions.

        Analyzes content and provides actionable recommendations
        for improving engagement based on:
        - Sentiment optimization
        - Length optimization
        - Readability improvement
        - Hashtag optimization
        - Timing recommendations

        Example:
    ```python
            optimizer = ContentOptimizer()
            result = optimizer.optimize(
                text="Check this out",
                platform="linkedin"
            )
            for suggestion in result.suggestions:
                print(f"{suggestion.type}: {suggestion.impact}")
    ```
    """

    def __init__(self) -> None:
        """Initialize content optimizer."""
        self.suggestion_generator = SuggestionGenerator()
        self.scorer = ContentScorer()

    def optimize(
        self,
        text: str,
        platform: str,
        analysis: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict] = None,
    ) -> OptimizationResult:
        """
        Generate optimization suggestions.

        Args:
            text: Content to optimize
            platform: Target platform
            analysis: Optional pre-computed analysis
            user_profile: Optional user engagement history

        Returns:
            Optimization recommendations

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Use provided analysis or create minimal one
        if analysis is None:
            analysis = {"features": {"hashtag_count": text.count("#")}}

        # Generate suggestions
        suggestions = self.suggestion_generator.generate(text, platform, analysis)

        # Calculate overall score
        overall_score = self.scorer.score(text, platform, analysis)

        # Estimate engagement lift
        predicted_lift = self._estimate_engagement_lift(overall_score, suggestions)

        # Determine best platform
        best_platform = self._determine_best_platform(text, analysis)

        # Generate rewrite examples
        rewrite_examples = self._generate_rewrites(text, suggestions)

        return OptimizationResult(
            overall_score=overall_score,
            suggestions=suggestions,
            predicted_engagement_lift=predicted_lift,
            best_platform=best_platform,
            best_time=None,  # Would integrate with timing module
            rewrite_examples=rewrite_examples,
        )

    def _estimate_engagement_lift(
        self, score: float, suggestions: List[ContentSuggestion]
    ) -> float:
        """Estimate potential engagement improvement."""
        # Simple heuristic: higher priority suggestions = more lift
        high_priority_count = sum(1 for s in suggestions if s.priority == "high")
        medium_priority_count = sum(1 for s in suggestions if s.priority == "medium")

        potential_lift = (high_priority_count * 15) + (medium_priority_count * 8)

        # Adjust based on current score
        if score > 80:
            potential_lift *= 0.5  # Less room for improvement

        return min(50.0, potential_lift)

    def _determine_best_platform(self, text: str, analysis: Dict[str, Any]) -> str:
        """Determine best platform for content."""
        text_length = len(text)

        # Simple heuristic based on length
        if text_length > 500:
            return "linkedin"
        elif text_length < 150:
            return "twitter"
        else:
            return "linkedin"

    def _generate_rewrites(
        self, text: str, suggestions: List[ContentSuggestion]
    ) -> List[str]:
        """Generate example rewrites."""
        rewrites = []

        # Generate 1-2 simple rewrite examples
        if any(s.type == "length" for s in suggestions):
            # Length-optimized version
            if len(text) > 200:
                rewrites.append(text[:197] + "...")
            else:
                rewrites.append(text + " [Add more context here]")

        if any(s.type == "hashtags" for s in suggestions):
            # Hashtag-optimized version
            if "#" not in text:
                rewrites.append(text + " #ContentMarketing #SocialMedia")

        return rewrites[:2]  # Max 2 examples
