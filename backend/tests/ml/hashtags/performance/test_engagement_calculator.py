"""Tests for engagement calculator."""

import pytest

from bufferiq.ml.hashtags.performance.engagement_calculator import EngagementCalculator


class TestEngagementCalculator:
    """Test EngagementCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return EngagementCalculator()

    def test_calculate_basic(self, calculator):
        """Test basic engagement calculation."""
        values = [100.0, 150.0, 120.0, 180.0, 140.0]
        metrics = calculator.calculate(values)

        assert isinstance(metrics, dict)
        assert "average" in metrics
        assert "median" in metrics
        assert "std" in metrics

    def test_calculate_average(self, calculator):
        """Test average calculation."""
        values = [100.0, 200.0]
        metrics = calculator.calculate(values)

        assert metrics["average"] == 150.0

    def test_calculate_median(self, calculator):
        """Test median calculation."""
        values = [100.0, 150.0, 200.0]
        metrics = calculator.calculate(values)

        assert metrics["median"] == 150.0

    def test_calculate_percentiles(self, calculator):
        """Test percentile calculation."""
        values = [100.0, 150.0, 200.0, 250.0]
        metrics = calculator.calculate(values)

        assert "percentile_25" in metrics
        assert "percentile_75" in metrics
        assert "percentile_90" in metrics

    def test_calculate_empty(self, calculator):
        """Test with empty values."""
        metrics = calculator.calculate([])

        assert metrics["average"] == 0.0
        assert metrics["median"] == 0.0
        assert metrics["std"] == 0.0

    def test_calculate_lift(self, calculator):
        """Test engagement lift calculation."""
        with_hashtag = [150.0, 160.0, 140.0]
        without_hashtag = [100.0, 110.0, 90.0]

        lift = calculator.calculate_lift(with_hashtag, without_hashtag)

        # (150 - 100) / 100 = 0.5 (50% lift)
        assert lift > 0
        assert isinstance(lift, float)

    def test_calculate_lift_zero_baseline(self, calculator):
        """Test lift with zero baseline."""
        with_hashtag = [150.0]
        without_hashtag = [0.0]

        lift = calculator.calculate_lift(with_hashtag, without_hashtag)

        assert lift == 0.0

    def test_calculate_lift_empty(self, calculator):
        """Test lift with empty lists."""
        lift = calculator.calculate_lift([], [])

        assert lift == 0.0


class TestROICalculator:
    """Test ROICalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        from bufferiq.ml.hashtags.performance.roi_calculator import ROICalculator
        return ROICalculator()

    def test_calculate_roi(self, calculator):
        """Test ROI calculation."""
        roi = calculator.calculate(
            avg_engagement_with=150.0,
            avg_engagement_without=120.0,
            hashtag="ai",
        )

        # (150 - 120) / (2 + 1) = 10.0
        assert roi == 10.0

    def test_calculate_efficiency_score(self, calculator):
        """Test efficiency score calculation."""
        score = calculator.calculate_efficiency_score(
            roi=5.0,
            hashtag_length=10,
        )

        assert 0 <= score <= 100