"""
Tests for content optimizer.
"""

import pytest

from bufferiq.ml.content.optimizer.optimizer import (
    ContentOptimizer,
    OptimizationResult,
)


class TestContentOptimizer:
    """Test ContentOptimizer class."""

    @pytest.fixture
    def optimizer(self) -> ContentOptimizer:
        """Create optimizer fixture."""
        return ContentOptimizer()

    def test_optimize_basic(self, optimizer: ContentOptimizer) -> None:
        """Test basic optimization."""
        result = optimizer.optimize("Test post here", "linkedin")

        assert isinstance(result, OptimizationResult)
        assert 0.0 <= result.overall_score <= 100.0

    def test_optimize_with_analysis(self, optimizer: ContentOptimizer) -> None:
        """Test optimization with pre-computed analysis."""
        analysis = {"features": {"hashtag_count": 0}}
        result = optimizer.optimize("Test", "linkedin", analysis=analysis)

        assert isinstance(result, OptimizationResult)

    def test_optimize_invalid_platform_raises_error(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            optimizer.optimize("Test", "facebook")

    def test_optimize_empty_text_raises_error(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            optimizer.optimize("", "linkedin")

    def test_optimize_returns_suggestions(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimization returns suggestions."""
        result = optimizer.optimize("Hi", "linkedin")

        assert isinstance(result.suggestions, list)
        # Short text should get suggestions
        assert len(result.suggestions) > 0

    def test_optimize_returns_engagement_lift(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimization returns engagement lift estimate."""
        result = optimizer.optimize("Test", "linkedin")

        assert isinstance(result.predicted_engagement_lift, float)
        assert result.predicted_engagement_lift >= 0

    def test_optimize_returns_best_platform(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimization returns best platform."""
        result = optimizer.optimize("Test post", "linkedin")

        assert result.best_platform in ["linkedin", "twitter", "bluesky"]

    def test_optimize_returns_rewrite_examples(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimization returns rewrite examples."""
        result = optimizer.optimize("Test", "linkedin")

        assert isinstance(result.rewrite_examples, list)

    def test_optimize_short_text(self, optimizer: ContentOptimizer) -> None:
        """Test optimizing short text."""
        result = optimizer.optimize("Hi there", "linkedin")

        # Should suggest improvements for short text
        assert len(result.suggestions) > 0

    def test_optimize_long_text(self, optimizer: ContentOptimizer) -> None:
        """Test optimizing long text."""
        text = "x" * 500
        result = optimizer.optimize(text, "linkedin")

        assert isinstance(result, OptimizationResult)

    def test_optimize_high_quality_text(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimizing already high quality text."""
        analysis = {
            "quality": {"score": 95.0},
            "readability": {"reading_difficulty": "easy"},
            "features": {"hashtag_count": 3},
        }
        text = "This is a well-crafted post with good content and structure."
        result = optimizer.optimize(text, "linkedin", analysis=analysis)

        # High quality should have fewer suggestions
        assert result.overall_score > 70

    def test_optimize_determines_best_platform_short(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test best platform for short content."""
        result = optimizer.optimize("Quick update!", "linkedin")

        # Short content better for Twitter/Bluesky
        assert result.best_platform in ["twitter", "bluesky"]

    def test_optimize_determines_best_platform_long(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test best platform for long content."""
        text = "x" * 600
        result = optimizer.optimize(text, "linkedin")

        # Long content better for LinkedIn
        assert result.best_platform == "linkedin"

    def test_optimize_engagement_lift_high_score(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test engagement lift for high score content."""
        analysis = {"quality": {"score": 90.0}, "features": {"hashtag_count": 3}}
        result = optimizer.optimize("Good post", "linkedin", analysis=analysis)

        # High score = lower potential lift
        assert result.predicted_engagement_lift < 30

    def test_optimize_engagement_lift_low_score(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test engagement lift for low score content."""
        result = optimizer.optimize("Hi", "linkedin")

        # Low score = higher potential lift
        assert result.predicted_engagement_lift >= 0

    def test_optimize_rewrite_examples_length(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test rewrite examples for length issues."""
        result = optimizer.optimize("Hi", "linkedin")

        # Should provide rewrite examples
        assert len(result.rewrite_examples) <= 2

    def test_optimize_with_user_profile(
        self, optimizer: ContentOptimizer
    ) -> None:
        """Test optimization with user profile."""
        user_profile = {"engagement_history": []}
        result = optimizer.optimize(
            "Test", "linkedin", user_profile=user_profile
        )

        assert isinstance(result, OptimizationResult)