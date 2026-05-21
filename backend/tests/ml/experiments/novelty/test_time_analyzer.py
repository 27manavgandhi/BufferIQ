"""Tests for time analyzer."""

from bufferiq.ml.experiments.novelty.time_analyzer import TimeAnalyzer


class TestTimeAnalyzer:
    """Test TimeAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = TimeAnalyzer()

    def test_analyze_increasing_trend(self):
        """Test increasing trend detection."""
        time_series = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        result = self.analyzer.analyze_trend(time_series)

        assert result["direction"] == "increasing"
        assert result["is_significant"] is True
        assert result["slope"] > 0

    def test_analyze_decreasing_trend(self):
        """Test decreasing trend detection."""
        time_series = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]

        result = self.analyzer.analyze_trend(time_series)

        assert result["direction"] == "decreasing"
        assert result["is_significant"] is True
        assert result["slope"] < 0

    def test_analyze_flat_trend(self):
        """Test flat trend detection."""
        time_series = [1.0, 1.01, 0.99, 1.0, 1.01, 0.99]

        result = self.analyzer.analyze_trend(time_series)

        assert result["direction"] == "flat"

    def test_detect_changepoint(self):
        """Test changepoint detection."""
        # Data with clear changepoint
        time_series = [1.0] * 10 + [2.0] * 10

        result = self.analyzer.detect_changepoint(time_series)

        assert result["has_changepoint"] is True
        assert 8 <= result["changepoint_index"] <= 12
