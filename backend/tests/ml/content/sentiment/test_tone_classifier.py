"""
Tests for tone classifier.
"""

import pytest

from bufferiq.ml.content.sentiment.tone_classifier import (
    ToneClassifier,
    Tone,
    ToneResult,
)


class TestToneClassifier:
    """Test ToneClassifier class."""

    @pytest.fixture
    def classifier(self) -> ToneClassifier:
        """Create tone classifier fixture."""
        return ToneClassifier()

    def test_classify_professional(self, classifier: ToneClassifier) -> None:
        """Test professional tone classification."""
        result = classifier.classify(
            "Please kindly review the document. Best regards."
        )

        assert result.tone == Tone.PROFESSIONAL

    def test_classify_casual(self, classifier: ToneClassifier) -> None:
        """Test casual tone classification."""
        result = classifier.classify("Hey! That's awesome, yeah!")

        assert result.tone == Tone.CASUAL

    def test_classify_urgent(self, classifier: ToneClassifier) -> None:
        """Test urgent tone classification."""
        result = classifier.classify(
            "URGENT: Please respond ASAP! Critical issue!"
        )

        assert result.tone == Tone.URGENT

    def test_classify_friendly(self, classifier: ToneClassifier) -> None:
        """Test friendly tone classification."""
        result = classifier.classify("Thanks so much! Really appreciate it!")

        assert result.tone == Tone.FRIENDLY

    def test_classify_returns_scores(self, classifier: ToneClassifier) -> None:
        """Test that classification returns score breakdown."""
        result = classifier.classify("Please review this.")

        assert Tone.PROFESSIONAL.value in result.scores
        assert Tone.CASUAL.value in result.scores

    def test_classify_confidence_range(
        self, classifier: ToneClassifier
    ) -> None:
        """Test confidence is in valid range."""
        result = classifier.classify("Please kindly review.")

        assert 0.0 <= result.confidence <= 1.0

    def test_classify_empty_text_raises_error(
        self, classifier: ToneClassifier
    ) -> None:
        """Test classifying empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            classifier.classify("")

    def test_classify_whitespace_only_raises_error(
        self, classifier: ToneClassifier
    ) -> None:
        """Test classifying whitespace-only text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            classifier.classify("   ")

    def test_classify_no_tone_indicators(
        self, classifier: ToneClassifier
    ) -> None:
        """Test text with no tone indicators defaults to professional."""
        result = classifier.classify("The report is complete.")

        assert result.tone == Tone.PROFESSIONAL
        assert result.confidence == 0.5

    def test_classify_mixed_tones(self, classifier: ToneClassifier) -> None:
        """Test text with mixed tones."""
        result = classifier.classify(
            "Hey, please kindly review ASAP. Thanks!"
        )

        # Should pick dominant tone
        assert isinstance(result.tone, Tone)

    def test_classify_case_insensitive(
        self, classifier: ToneClassifier
    ) -> None:
        """Test classification is case-insensitive."""
        result = classifier.classify("PLEASE KINDLY REVIEW")

        assert result.tone == Tone.PROFESSIONAL

    def test_classify_strong_professional(
        self, classifier: ToneClassifier
    ) -> None:
        """Test strong professional tone."""
        result = classifier.classify(
            "Dear Sir/Madam, I respectfully request your consideration. "
            "Kindly provide feedback at your earliest convenience. "
            "Sincerely, John"
        )

        assert result.tone == Tone.PROFESSIONAL

    def test_classify_strong_casual(self, classifier: ToneClassifier) -> None:
        """Test strong casual tone."""
        result = classifier.classify(
            "Hey dude! Yeah that's super cool! "
            "Wanna grab coffee? Awesome!"
        )

        assert result.tone == Tone.CASUAL

    def test_classify_strong_urgent(self, classifier: ToneClassifier) -> None:
        """Test strong urgent tone."""
        result = classifier.classify(
            "URGENT URGENT URGENT! Need this ASAP immediately! "
            "Critical situation requires quick action now!"
        )

        assert result.tone == Tone.URGENT

    def test_classify_short_text(self, classifier: ToneClassifier) -> None:
        """Test classification on short text."""
        result = classifier.classify("Thanks!")

        assert result.tone == Tone.FRIENDLY

    def test_classify_long_text(self, classifier: ToneClassifier) -> None:
        """Test classification on longer text."""
        text = (
            "I wanted to reach out and thank you for all your help. "
            "Your support has been wonderful and I really appreciate "
            "everything you've done. Looking forward to working together!"
        )
        result = classifier.classify(text)

        assert result.tone == Tone.FRIENDLY

    def test_classify_multiple_indicators(
        self, classifier: ToneClassifier
    ) -> None:
        """Test text with multiple tone indicators."""
        result = classifier.classify(
            "Please, please, please respond kindly and professionally."
        )

        assert result.tone == Tone.PROFESSIONAL

    def test_classify_scores_valid(self, classifier: ToneClassifier) -> None:
        """Test that all scores are valid."""
        result = classifier.classify("Thanks for your help!")

        for score in result.scores.values():
            assert 0.0 <= score <= 1.0