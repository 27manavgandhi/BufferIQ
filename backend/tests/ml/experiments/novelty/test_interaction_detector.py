"""Tests for interaction detector."""

from bufferiq.ml.experiments.novelty.interaction_detector import InteractionDetector


class TestInteractionDetector:
    """Test InteractionDetector."""

    def setup_method(self):
        """Setup test."""
        self.detector = InteractionDetector()

    def test_detect_interaction(self):
        """Test interaction detection."""
        # Treatment effect varies by segment
        result = self.detector.detect_interaction(
            segment_treatment_means={"new": 0.12, "old": 0.06},
            segment_control_means={"new": 0.05, "old": 0.05},
            segment_sizes={"new": 1000, "old": 1000},
        )

        assert result["has_interaction"] is True
        assert "segment_lifts" in result

    def test_no_interaction(self):
        """Test no interaction."""
        # Treatment effect consistent across segments
        result = self.detector.detect_interaction(
            segment_treatment_means={"new": 0.10, "old": 0.10},
            segment_control_means={"new": 0.05, "old": 0.05},
            segment_sizes={"new": 1000, "old": 1000},
        )

        assert result["has_interaction"] is False

    def test_detect_time_interaction(self):
        """Test time-based interaction."""
        # Effect decreasing over time
        daily_effects = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]

        result = self.detector.detect_time_interaction(daily_effects)

        assert result["has_time_interaction"] is True
        assert result["trend_direction"] == "decreasing"

    def test_detect_platform_interaction(self):
        """Test platform interaction."""
        result = self.detector.detect_platform_interaction(
            platform_treatment_means={"linkedin": 0.10, "twitter": 0.06},
            platform_control_means={"linkedin": 0.05, "twitter": 0.05},
            platform_sizes={"linkedin": 1000, "twitter": 1000},
        )

        assert "has_interaction" in result
