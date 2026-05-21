"""Tests for segmentation analyzer."""

from bufferiq.ml.experiments.results.segmentation import SegmentationAnalyzer
from bufferiq.ml.experiments.design.designer import MetricType


class TestSegmentationAnalyzer:
    """Test SegmentationAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = SegmentationAnalyzer()

    def test_analyze_by_segments(self):
        """Test segment analysis."""
        results = self.analyzer.analyze_by_segments(
            control_data_by_segment={
                "new": [0, 1, 0, 1, 0] * 20,
                "returning": [1, 1, 1, 0, 1] * 20,
            },
            treatment_data_by_segment={
                "new": [1, 1, 1, 1, 0] * 20,
                "returning": [1, 1, 0, 1, 1] * 20,
            },
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        assert "new" in results
        assert "returning" in results
        assert results["new"]["status"] == "analyzed"

    def test_identify_best_segments(self):
        """Test identifying best segments."""
        segment_results = {
            "new": {
                "status": "analyzed",
                "relative_diff": 0.20,
                "is_significant": True,
                "p_value": 0.01,
            },
            "returning": {
                "status": "analyzed",
                "relative_diff": 0.05,
                "is_significant": False,
                "p_value": 0.50,
            },
        }

        best = self.analyzer.identify_best_segments(segment_results, top_n=1)

        assert len(best) == 1
        assert best[0]["segment"] == "new"

    def test_calculate_hte(self):
        """Test heterogeneous treatment effects."""
        segment_results = {
            "seg1": {
                "status": "analyzed",
                "relative_diff": 0.10,
            },
            "seg2": {
                "status": "analyzed",
                "relative_diff": 0.20,
            },
        }

        hte = self.analyzer.calculate_heterogeneous_treatment_effects(
            segment_results
        )

        assert "has_hte" in hte
        assert "mean_effect" in hte
