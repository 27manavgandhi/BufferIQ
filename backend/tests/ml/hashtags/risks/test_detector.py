"""Tests for risk detector."""

import pytest

from bufferiq.ml.hashtags.risks.detector import (
    HashtagRiskDetector,
    HashtagRisk,
)


class TestHashtagRiskDetector:
    """Test HashtagRiskDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return HashtagRiskDetector()

    def test_assess_safe_hashtag(self, detector):
        """Test assessing safe hashtag."""
        risk = detector.assess(
            hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(risk, HashtagRisk)
        assert risk.hashtag == "ai"
        assert risk.risk_level in ["none", "low"]

    def test_assess_banned_hashtag(self, detector):
        """Test assessing banned hashtag."""
        risk = detector.assess(
            hashtag="spam",  # In default banned list
            platform="linkedin",
        )

        assert risk.is_banned
        assert risk.risk_level in ["high", "critical"]
        assert risk.recommendation == "avoid"

    def test_assess_spam_pattern(self, detector):
        """Test spam pattern detection."""
        risk = detector.assess(
            hashtag="followback",
            platform="linkedin",
        )

        assert risk.is_spam or risk.is_banned
        assert risk.recommendation == "avoid"

    def test_assess_invalid_platform(self, detector):
        """Test with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            detector.assess(
                hashtag="ai",
                platform="facebook",
            )

    def test_risk_reasons(self, detector):
        """Test risk reasons are provided."""
        risk = detector.assess(
            hashtag="spam",
            platform="linkedin",
        )

        assert len(risk.risk_reasons) > 0
        assert all(isinstance(r, str) for r in risk.risk_reasons)

    def test_alternatives_for_risky(self, detector):
        """Test alternatives provided for risky hashtags."""
        risk = detector.assess(
            hashtag="followback",
            platform="linkedin",
        )

        if risk.risk_level not in ["none", "low"]:
            # Should have alternatives
            assert isinstance(risk.alternatives, list)

    def test_assess_nsfw(self, detector):
        """Test NSFW detection."""
        risk = detector.assess(
            hashtag="nsfw",
            platform="linkedin",
        )

        assert risk.is_nsfw
        assert risk.risk_level in ["high", "critical"]

    def test_recommendation_levels(self, detector):
        """Test recommendation matches risk level."""
        # Safe hashtag
        safe_risk = detector.assess("ai", "linkedin")
        assert safe_risk.recommendation == "use"

        # Risky hashtag
        risky_risk = detector.assess("spam", "linkedin")
        assert risky_risk.recommendation == "avoid"


class TestSafetyChecker:
    """Test SafetyChecker class."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        from bufferiq.ml.hashtags.risks.safety_checker import SafetyChecker
        return SafetyChecker()

    def test_is_brand_safe(self, checker):
        """Test brand safety check."""
        is_safe = checker.is_brand_safe(
            hashtag="ai",
            brand_guidelines=["technology", "professional"],
        )

        assert isinstance(is_safe, bool)

    def test_unsafe_keywords(self, checker):
        """Test unsafe keyword detection."""
        is_safe = checker.is_brand_safe(
            hashtag="hate",
        )

        assert not is_safe

    def test_get_safety_score(self, checker):
        """Test safety score calculation."""
        score = checker.get_safety_score(
            hashtag="ai",
            brand_guidelines=["technology"],
        )

        assert 0 <= score <= 100

    def test_safety_score_with_unsafe(self, checker):
        """Test safety score for unsafe hashtag."""
        score = checker.get_safety_score(
            hashtag="violence",
        )

        assert score < 50  # Should be low