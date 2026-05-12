"""
Tests for emotion detector.
"""

import pytest

from bufferiq.ml.content.sentiment.emotion_detector import (
    EmotionDetector,
    Emotion,
    EmotionResult,
)


class TestEmotionDetector:
    """Test EmotionDetector class."""

    @pytest.fixture
    def detector(self) -> EmotionDetector:
        """Create emotion detector fixture."""
        return EmotionDetector()

    def test_detect_joy(self, detector: EmotionDetector) -> None:
        """Test joy emotion detection."""
        result = detector.detect("I'm so happy and excited!")

        assert result.emotion == Emotion.JOY
        assert result.confidence > 0.0

    def test_detect_anger(self, detector: EmotionDetector) -> None:
        """Test anger emotion detection."""
        result = detector.detect("I'm so angry and furious!")

        assert result.emotion == Emotion.ANGER

    def test_detect_sadness(self, detector: EmotionDetector) -> None:
        """Test sadness emotion detection."""
        result = detector.detect("I'm so sad and disappointed.")

        assert result.emotion == Emotion.SADNESS

    def test_detect_fear(self, detector: EmotionDetector) -> None:
        """Test fear emotion detection."""
        result = detector.detect("I'm scared and afraid.")

        assert result.emotion == Emotion.FEAR

    def test_detect_surprise(self, detector: EmotionDetector) -> None:
        """Test surprise emotion detection."""
        result = detector.detect("Wow! I'm so surprised!")

        assert result.emotion == Emotion.SURPRISE

    def test_detect_neutral(self, detector: EmotionDetector) -> None:
        """Test neutral emotion detection."""
        result = detector.detect("This is a factual statement.")

        assert result.emotion == Emotion.NEUTRAL

    def test_detect_returns_scores(self, detector: EmotionDetector) -> None:
        """Test that detection returns score breakdown."""
        result = detector.detect("Happy!")

        assert Emotion.JOY.value in result.scores
        assert Emotion.ANGER.value in result.scores
        assert Emotion.SADNESS.value in result.scores

    def test_detect_confidence_range(self, detector: EmotionDetector) -> None:
        """Test confidence is in valid range."""
        result = detector.detect("Very happy!")

        assert 0.0 <= result.confidence <= 1.0

    def test_detect_empty_text_raises_error(
        self, detector: EmotionDetector
    ) -> None:
        """Test detecting empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            detector.detect("")

    def test_detect_whitespace_only_raises_error(
        self, detector: EmotionDetector
    ) -> None:
        """Test detecting whitespace-only text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            detector.detect("   ")

    def test_detect_multiple_emotions(
        self, detector: EmotionDetector
    ) -> None:
        """Test text with multiple emotions."""
        result = detector.detect("I'm happy but also worried.")

        # Should detect dominant emotion
        assert isinstance(result.emotion, Emotion)

    def test_detect_strong_joy(self, detector: EmotionDetector) -> None:
        """Test strong joy emotion."""
        result = detector.detect(
            "I'm absolutely thrilled and delighted! So happy and joyful!"
        )

        assert result.emotion == Emotion.JOY
        assert result.confidence > 0.5

    def test_detect_case_insensitive(self, detector: EmotionDetector) -> None:
        """Test detection is case-insensitive."""
        result = detector.detect("HAPPY AND EXCITED")

        assert result.emotion == Emotion.JOY

    def test_detect_with_punctuation(
        self, detector: EmotionDetector
    ) -> None:
        """Test detection with punctuation."""
        result = detector.detect("Happy! Excited! Joyful!")

        assert result.emotion == Emotion.JOY

    def test_detect_no_emotion_keywords(
        self, detector: EmotionDetector
    ) -> None:
        """Test text with no emotion keywords."""
        result = detector.detect("The meeting is scheduled for tomorrow.")

        assert result.emotion == Emotion.NEUTRAL
        assert result.confidence == 1.0

    def test_detect_mixed_positive_negative(
        self, detector: EmotionDetector
    ) -> None:
        """Test mixed positive and negative emotions."""
        result = detector.detect("I'm happy but also angry.")

        # Should pick dominant or return one
        assert isinstance(result.emotion, Emotion)

    def test_detect_short_text(self, detector: EmotionDetector) -> None:
        """Test detection on short text."""
        result = detector.detect("Happy!")

        assert result.emotion == Emotion.JOY

    def test_detect_long_text(self, detector: EmotionDetector) -> None:
        """Test detection on longer text."""
        text = (
            "I am feeling wonderful today. Everything is going great "
            "and I couldn't be happier. This is truly an amazing experience."
        )
        result = detector.detect(text)

        assert result.emotion == Emotion.JOY

    def test_detect_scores_sum_valid(self, detector: EmotionDetector) -> None:
        """Test that scores are valid."""
        result = detector.detect("Happy and excited!")

        # Scores should be valid probabilities
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0