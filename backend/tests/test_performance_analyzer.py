"""Tests for performance analyzer."""

import numpy as np
import pandas as pd
import pytest

from bufferiq.ml.evaluation.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Test performance analyzer."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data."""
        np.random.seed(42)
        y_true = np.random.rand(100) * 10
        y_pred = y_true + np.random.randn(100) * 0.5
        return y_true, y_pred

    @pytest.fixture
    def analyzer(self) -> PerformanceAnalyzer:
        """Create analyzer."""
        return PerformanceAnalyzer()

    def test_init(self, analyzer: PerformanceAnalyzer) -> None:
        """Test initialization."""
        assert analyzer is not None

    def test_analyze_performance_by_percentile(
        self, analyzer: PerformanceAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test percentile analysis."""
        y_true, y_pred = sample_data

        percentile_perf = analyzer.analyze_performance_by_percentile(
            y_true, y_pred, [25, 50, 75, 90]
        )

        assert len(percentile_perf) > 0
        assert "percentile" in percentile_perf.columns
        assert "mae" in percentile_perf.columns

    def test_analyze_performance_by_prediction_confidence(
        self, analyzer: PerformanceAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test confidence analysis."""
        y_true, y_pred = sample_data

        confidence_perf = analyzer.analyze_performance_by_prediction_confidence(
            y_true, y_pred
        )

        assert len(confidence_perf) > 0
        assert "confidence_level" in confidence_perf.columns

    def test_detect_systematic_bias(
        self, analyzer: PerformanceAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test bias detection."""
        y_true, y_pred = sample_data

        bias = analyzer.detect_systematic_bias(y_true, y_pred)

        assert "overall_bias" in bias
        assert "overestimation_rate" in bias
        assert "underestimation_rate" in bias
        assert "bias_by_range" in bias

    def test_analyze_error_correlation_with_features(
        self, analyzer: PerformanceAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test error correlation analysis."""
        y_true, y_pred = sample_data
        errors = y_true - y_pred

        features = pd.DataFrame(
            {
                "f1": np.random.rand(100),
                "f2": np.random.rand(100),
                "f3": np.random.rand(100),
            }
        )

        error_corr = analyzer.analyze_error_correlation_with_features(
            errors, features, top_n=3
        )

        assert len(error_corr) <= 3
        if len(error_corr) > 0:
            assert "feature" in error_corr.columns
            assert "correlation" in error_corr.columns
