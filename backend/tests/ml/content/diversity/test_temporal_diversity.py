"""
Tests for temporal diversity analyzer.
"""

from datetime import datetime, timedelta

import pytest

from bufferiq.ml.content.diversity.temporal_diversity import (
    TemporalDiversityAnalyzer,
)


class TestTemporalDiversityAnalyzer:
    """Test TemporalDiversityAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> TemporalDiversityAnalyzer:
        """Create analyzer fixture."""
        return TemporalDiversityAnalyzer()

    @pytest.fixture
    def sample_timestamps(self) -> list:
        """Create sample timestamps."""
        base = datetime(2024, 1, 1, 9, 0)
        return [
            base,
            base + timedelta(hours=3),
            base + timedelta(hours=6),
            base + timedelta(days=1),
        ]

    def test_calculate_diversity_basic(
        self, analyzer: TemporalDiversityAnalyzer, sample_timestamps: list
    ) -> None:
        """Test basic diversity calculation."""
        diversity = analyzer.calculate_diversity(sample_timestamps)

        assert 0.0 <= diversity <= 1.0

    def test_calculate_diversity_high(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test high temporal diversity."""
        # Different hours across the day
        timestamps = [
            datetime(2024, 1, 1, h, 0) for h in [1, 6, 12, 18, 23]
        ]
        diversity = analyzer.calculate_diversity(timestamps)

        assert diversity > 0.3

    def test_calculate_diversity_low(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test low temporal diversity."""
        # All same hour
        timestamps = [datetime(2024, 1, 1, 9, i) for i in range(5)]
        diversity = analyzer.calculate_diversity(timestamps)

        assert diversity == 0.0

    def test_calculate_diversity_empty_raises_error(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test empty timestamps raises error."""
        with pytest.raises(ValueError, match="Timestamps list cannot be empty"):
            analyzer.calculate_diversity([])

    def test_calculate_diversity_single_timestamp(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test single timestamp returns zero."""
        timestamps = [datetime(2024, 1, 1, 9, 0)]
        diversity = analyzer.calculate_diversity(timestamps)

        assert diversity == 0.0

    def test_calculate_weekday_diversity(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test weekday diversity calculation."""
        # Different days of week
        timestamps = [
            datetime(2024, 1, 1 + i, 9, 0) for i in range(7)
        ]
        diversity = analyzer.calculate_weekday_diversity(timestamps)

        assert 0.0 <= diversity <= 1.0

    def test_calculate_weekday_diversity_high(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test high weekday diversity."""
        # All different weekdays
        timestamps = [
            datetime(2024, 1, 1 + i, 9, 0) for i in range(7)
        ]
        diversity = analyzer.calculate_weekday_diversity(timestamps)

        assert diversity > 0.9

    def test_calculate_weekday_diversity_low(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test low weekday diversity."""
        # All Monday
        timestamps = [
            datetime(2024, 1, 1 + i * 7, 9, 0) for i in range(5)
        ]
        diversity = analyzer.calculate_weekday_diversity(timestamps)

        assert diversity == 0.0

    def test_calculate_weekday_diversity_empty_raises_error(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test empty timestamps raises error."""
        with pytest.raises(ValueError, match="Timestamps list cannot be empty"):
            analyzer.calculate_weekday_diversity([])

    def test_calculate_time_intervals(
        self, analyzer: TemporalDiversityAnalyzer, sample_timestamps: list
    ) -> None:
        """Test time interval calculation."""
        intervals = analyzer.calculate_time_intervals(sample_timestamps)

        assert isinstance(intervals, list)
        assert len(intervals) == len(sample_timestamps) - 1
        assert all(i > 0 for i in intervals)

    def test_calculate_time_intervals_empty_raises_error(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test empty timestamps raises error."""
        with pytest.raises(ValueError, match="Timestamps list cannot be empty"):
            analyzer.calculate_time_intervals([])

    def test_calculate_time_intervals_single_timestamp(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test single timestamp returns empty list."""
        timestamps = [datetime(2024, 1, 1, 9, 0)]
        intervals = analyzer.calculate_time_intervals(timestamps)

        assert intervals == []

    def test_calculate_time_intervals_hours(
        self, analyzer: TemporalDiversityAnalyzer
    ) -> None:
        """Test intervals are in hours."""
        base = datetime(2024, 1, 1, 9, 0)
        timestamps = [base, base + timedelta(hours=2)]
        intervals = analyzer.calculate_time_intervals(timestamps)

        assert len(intervals) == 1
        assert abs(intervals[0] - 2.0) < 0.01  # Should be ~2 hours