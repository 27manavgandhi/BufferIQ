"""Tests for error analyzer."""

import numpy as np
import pandas as pd
import pytest

from bufferiq.ml.evaluation.error_analyzer import ErrorAnalyzer


class TestErrorAnalyzer:
    """Test error analyzer."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data."""
        np.random.seed(42)
        y_true = np.random.rand(100) * 10
        y_pred = y_true + np.random.randn(100) * 0.5
        return y_true, y_pred

    @pytest.fixture
    def analyzer(self) -> ErrorAnalyzer:
        """Create analyzer."""
        return ErrorAnalyzer()

    def test_init(self, analyzer: ErrorAnalyzer) -> None:
        """Test initialization."""
        assert analyzer is not None

    def test_classify_errors(
        self, analyzer: ErrorAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test error classification."""
        y_true, y_pred = sample_data

        error_classes = analyzer.classify_errors(y_true, y_pred)

        assert "low_error" in error_classes
        assert "medium_error" in error_classes
        assert "high_error" in error_classes
        assert "very_high_error" in error_classes
        assert sum(error_classes.values()) == len(y_true)

    def test_identify_failure_modes(
        self, analyzer: ErrorAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test failure mode identification."""
        y_true, y_pred = sample_data

        features = pd.DataFrame(
            {
                "text_length": np.random.randint(50, 300, 100),
                "hashtag_count": np.random.randint(0, 5, 100),
            }
        )

        failure_modes = analyzer.identify_failure_modes(
            pd.Series(y_true), y_pred, features, error_threshold=0.5
        )

        assert isinstance(failure_modes, list)

    def test_analyze_error_by_feature_ranges(
        self, analyzer: ErrorAnalyzer, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test error by feature range analysis."""
        y_true, y_pred = sample_data
        errors = y_true - y_pred

        features = pd.DataFrame(
            {
                "text_length": np.random.randint(50, 300, 100),
            }
        )

        error_by_range = analyzer.analyze_error_by_feature_ranges(
            errors, features, "text_length", n_bins=5
        )

        assert len(error_by_range) > 0
        assert "range" in error_by_range.columns
        assert "mean_abs_error" in error_by_range.columns