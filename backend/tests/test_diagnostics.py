"""Tests for model diagnostics."""

import numpy as np
import pandas as pd
import pytest

from bufferiq.ml.evaluation.diagnostics import ModelDiagnostics


class TestModelDiagnostics:
    """Test model diagnostics."""

    @pytest.fixture
    def diagnostics(self) -> ModelDiagnostics:
        """Create diagnostics."""
        return ModelDiagnostics()

    def test_init(self, diagnostics: ModelDiagnostics) -> None:
        """Test initialization."""
        assert diagnostics is not None

    def test_check_overfitting(self, diagnostics: ModelDiagnostics) -> None:
        """Test overfitting check."""
        train_metrics = {"r2": 0.95, "mae": 0.05}
        val_metrics = {"r2": 0.75, "mae": 0.15}

        result = diagnostics.check_overfitting(train_metrics, val_metrics, threshold=0.1)

        assert "is_overfitting" in result
        assert "train_val_gap" in result
        assert "severity" in result
        assert result["is_overfitting"]

    def test_check_overfitting_no_overfit(
        self, diagnostics: ModelDiagnostics
    ) -> None:
        """Test no overfitting case."""
        train_metrics = {"r2": 0.80, "mae": 0.10}
        val_metrics = {"r2": 0.78, "mae": 0.11}

        result = diagnostics.check_overfitting(train_metrics, val_metrics)

        assert not result["is_overfitting"]

    def test_check_underfitting(self, diagnostics: ModelDiagnostics) -> None:
        """Test underfitting check."""
        metrics = {"r2": 0.40, "mae": 0.25}

        result = diagnostics.check_underfitting(metrics, min_r2=0.5)

        assert "is_underfitting" in result
        assert "severity" in result
        assert result["is_underfitting"]

    def test_check_underfitting_good_fit(
        self, diagnostics: ModelDiagnostics
    ) -> None:
        """Test good fit case."""
        metrics = {"r2": 0.75, "mae": 0.12}

        result = diagnostics.check_underfitting(metrics, min_r2=0.5)

        assert not result["is_underfitting"]

    def test_check_residual_patterns(self, diagnostics: ModelDiagnostics) -> None:
        """Test residual pattern check."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        result = diagnostics.check_residual_patterns(residuals)

        assert "mean_residual" in result
        assert "mean_close_to_zero" in result
        assert "constant_variance" in result
        assert "is_normal" in result

    def test_check_feature_importance_concentration(
        self, diagnostics: ModelDiagnostics
    ) -> None:
        """Test feature importance concentration check."""
        importance = pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(10)],
                "importance": [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],
            }
        )

        result = diagnostics.check_feature_importance_concentration(
            importance, threshold=0.8
        )

        assert "is_concentrated" in result
        assert "features_to_threshold" in result