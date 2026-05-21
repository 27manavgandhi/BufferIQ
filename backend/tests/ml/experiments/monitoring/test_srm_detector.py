"""Tests for SRM detector."""

from bufferiq.ml.experiments.monitoring.srm_detector import SRMDetector


class TestSRMDetector:
    """Test SRMDetector."""

    def setup_method(self):
        """Setup test."""
        self.detector = SRMDetector(alpha=0.001)

    def test_detect_srm_present(self):
        """Test SRM detection when present."""
        result = self.detector.detect_srm(
            variant_counts={"control": 1200, "treatment": 800},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        assert result["has_srm"] is True
        assert result["chi2_p_value"] < 0.001

    def test_detect_srm_absent(self):
        """Test no SRM when balanced."""
        result = self.detector.detect_srm(
            variant_counts={"control": 1000, "treatment": 1000},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        assert result["has_srm"] is False

    def test_calculate_deviations(self):
        """Test deviation calculation."""
        result = self.detector.detect_srm(
            variant_counts={"control": 1100, "treatment": 900},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        assert "deviations" in result
        assert "control" in result["deviations"]
