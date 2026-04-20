"""Tests for temporal analyzer."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from bufferiq.ml.analysis.temporal_analyzer import TemporalAnalyzer


@pytest.fixture
def temporal_df():
    """Create sample DataFrame with temporal data."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(30)]

    return pd.DataFrame(
        {
            "published_at": dates,
            "date": [d.date() for d in dates],
            "hour": [d.hour for d in dates],
            "day_of_week": [d.weekday() for d in dates],
            "day_name": [d.strftime("%A") for d in dates],
            "month": [d.month for d in dates],
            "is_weekend": [d.weekday() >= 5 for d in dates],
            "engagement_rate": [0.05 + i * 0.001 for i in range(30)],
        }
    )


def test_hourly_patterns(temporal_df):
    """Test hourly pattern analysis."""
    analyzer = TemporalAnalyzer()
    hourly = analyzer.hourly_patterns(temporal_df, "engagement_rate")

    assert isinstance(hourly, pd.DataFrame)
    assert "hour" in hourly.columns
    assert "mean" in hourly.columns
    assert "count" in hourly.columns
    assert "ci_lower" in hourly.columns
    assert "ci_upper" in hourly.columns


def test_hourly_patterns_missing_column():
    """Test hourly patterns with missing column."""
    analyzer = TemporalAnalyzer()
    df = pd.DataFrame({"engagement_rate": [0.1, 0.2]})

    with pytest.raises(ValueError, match="hour"):
        analyzer.hourly_patterns(df, "engagement_rate")


def test_daily_patterns(temporal_df):
    """Test daily pattern analysis."""
    analyzer = TemporalAnalyzer()
    daily = analyzer.daily_patterns(temporal_df, "engagement_rate")

    assert isinstance(daily, pd.DataFrame)
    assert "day_of_week" in daily.columns
    assert "day_name" in daily.columns
    assert "mean" in daily.columns
    assert "count" in daily.columns


def test_weekly_trends(temporal_df):
    """Test weekly trend analysis."""
    analyzer = TemporalAnalyzer()
    weekly = analyzer.weekly_trends(temporal_df, "engagement_rate")

    assert isinstance(weekly, pd.DataFrame)
    assert "week_start" in weekly.columns
    assert "mean" in weekly.columns
    assert "count" in weekly.columns


def test_seasonal_patterns(temporal_df):
    """Test seasonal pattern analysis."""
    analyzer = TemporalAnalyzer()
    seasonal = analyzer.seasonal_patterns(temporal_df, "engagement_rate")

    assert isinstance(seasonal, dict)
    assert "monthly_means" in seasonal
    assert "peak_month" in seasonal
    assert "low_month" in seasonal


def test_optimal_posting_windows(temporal_df):
    """Test optimal posting window identification."""
    analyzer = TemporalAnalyzer()
    windows = analyzer.optimal_posting_windows(temporal_df, top_n=3)

    assert isinstance(windows, list)
    assert len(windows) <= 3
    for window in windows:
        assert "day_of_week" in window
        assert "day_name" in window
        assert "hour" in window
        assert "mean_engagement" in window


def test_optimal_posting_windows_with_platform(temporal_df):
    """Test optimal posting windows filtered by platform."""
    temporal_df["platform"] = "linkedin"
    analyzer = TemporalAnalyzer()
    windows = analyzer.optimal_posting_windows(
        temporal_df, platform="linkedin", top_n=3
    )

    assert isinstance(windows, list)


def test_optimal_posting_windows_insufficient_data():
    """Test optimal posting windows with insufficient data."""
    analyzer = TemporalAnalyzer()
    df = pd.DataFrame(
        {
            "day_of_week": [0, 1],
            "day_name": ["Monday", "Tuesday"],
            "hour": [10, 11],
            "engagement_rate": [0.05, 0.06],
        }
    )

    windows = analyzer.optimal_posting_windows(df, top_n=5)
    assert len(windows) == 0  # Not enough posts per window


def test_weekend_vs_weekday(temporal_df):
    """Test weekend vs weekday comparison."""
    analyzer = TemporalAnalyzer()
    comparison = analyzer.weekend_vs_weekday(temporal_df, "engagement_rate")

    assert isinstance(comparison, dict)
    assert "weekend_mean" in comparison
    assert "weekday_mean" in comparison
    assert "weekend_count" in comparison
    assert "weekday_count" in comparison


def test_weekend_vs_weekday_missing_column():
    """Test weekend vs weekday without is_weekend column."""
    analyzer = TemporalAnalyzer()
    df = pd.DataFrame({"engagement_rate": [0.1, 0.2]})

    with pytest.raises(ValueError, match="is_weekend"):
        analyzer.weekend_vs_weekday(df, "engagement_rate")
