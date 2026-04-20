"""Tests for engagement analyzer."""

import pytest
import pandas as pd
import numpy as np

from bufferiq.ml.analysis.engagement_analyzer import EngagementAnalyzer


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "content": ["Post 1", "Post 2", "Post 3", "Post 4", "Post 5"],
            "likes": [10, 20, 30, 40, 50],
            "comments": [1, 2, 3, 4, 5],
            "shares": [0, 1, 2, 3, 4],
            "impressions": [100, 200, 300, 400, 500],
            "platform": ["linkedin", "twitter", "linkedin", "twitter", "linkedin"],
        }
    )


def test_calculate_engagement_rate(sample_df):
    """Test engagement rate calculation."""
    analyzer = EngagementAnalyzer()
    result = analyzer.calculate_engagement_rate(sample_df)

    assert "engagement_rate" in result.columns
    assert "total_engagement" in result.columns
    assert len(result) == len(sample_df)
    assert (result["engagement_rate"] >= 0).all()
    assert (result["engagement_rate"] <= 1).all()


def test_calculate_engagement_rate_zero_impressions():
    """Test engagement rate with zero impressions."""
    analyzer = EngagementAnalyzer()
    df = pd.DataFrame(
        {
            "likes": [10],
            "comments": [2],
            "shares": [1],
            "impressions": [0],
        }
    )

    result = analyzer.calculate_engagement_rate(df)
    assert result["engagement_rate"].iloc[0] == 0.0


def test_analyze_distribution(sample_df):
    """Test distribution analysis."""
    analyzer = EngagementAnalyzer()
    df = analyzer.calculate_engagement_rate(sample_df)
    stats = analyzer.analyze_distribution(df, "engagement_rate")

    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "count" in stats
    assert stats["count"] == len(df)


def test_analyze_distribution_missing_column(sample_df):
    """Test distribution analysis with missing column."""
    analyzer = EngagementAnalyzer()

    with pytest.raises(ValueError, match="not found"):
        analyzer.analyze_distribution(sample_df, "nonexistent_column")


def test_identify_outliers_iqr(sample_df):
    """Test outlier identification using IQR method."""
    analyzer = EngagementAnalyzer()
    df = analyzer.calculate_engagement_rate(sample_df)
    outliers = analyzer.identify_outliers(df, "likes", method="iqr")

    assert isinstance(outliers, pd.DataFrame)


def test_identify_outliers_zscore(sample_df):
    """Test outlier identification using Z-score method."""
    analyzer = EngagementAnalyzer()
    outliers = analyzer.identify_outliers(sample_df, "likes", method="zscore")

    assert isinstance(outliers, pd.DataFrame)


def test_identify_outliers_invalid_method(sample_df):
    """Test outlier identification with invalid method."""
    analyzer = EngagementAnalyzer()

    with pytest.raises(ValueError, match="Invalid method"):
        analyzer.identify_outliers(sample_df, "likes", method="invalid")


def test_calculate_correlations(sample_df):
    """Test correlation calculation."""
    analyzer = EngagementAnalyzer()
    corr = analyzer.calculate_correlations(sample_df)

    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]  # Square matrix


def test_find_strong_correlations(sample_df):
    """Test finding strong correlations."""
    analyzer = EngagementAnalyzer()
    corr = analyzer.calculate_correlations(sample_df)
    strong = analyzer.find_strong_correlations(corr, threshold=0.5)

    assert isinstance(strong, list)
    for feat1, feat2, corr_val in strong:
        assert isinstance(feat1, str)
        assert isinstance(feat2, str)
        assert isinstance(corr_val, float)
        assert abs(corr_val) >= 0.5


def test_platform_comparison(sample_df):
    """Test platform comparison."""
    analyzer = EngagementAnalyzer()
    df = analyzer.calculate_engagement_rate(sample_df)
    comparison = analyzer.platform_comparison(df, "engagement_rate")

    assert "means" in comparison
    assert "medians" in comparison
    assert "counts" in comparison
    assert isinstance(comparison["means"], dict)


def test_platform_comparison_missing_column():
    """Test platform comparison without platform column."""
    analyzer = EngagementAnalyzer()
    df = pd.DataFrame({"engagement_rate": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="platform"):
        analyzer.platform_comparison(df, "engagement_rate")


def test_segment_by_performance(sample_df):
    """Test performance segmentation."""
    analyzer = EngagementAnalyzer()
    df = analyzer.calculate_engagement_rate(sample_df)
    result = analyzer.segment_by_performance(df, "engagement_rate", n_segments=3)

    assert "performance_segment" in result.columns
    assert "performance_label" in result.columns
    assert result["performance_label"].isin(["low", "medium", "high"]).all()
