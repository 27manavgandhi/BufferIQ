"""Tests for feature selector."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bufferiq.ml.features.selector import FeatureSelector


class TestFeatureSelector:
    """Test feature selector."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [
                    2,
                    4,
                    6,
                    8,
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                ],  # Correlated with feature1
                "feature3": [
                    10,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                    45,
                    50,
                    55,
                ],  # Correlated with y
                "feature4": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # Low variance
                "feature5": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            }
        )
        y = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        return X, y

    def test_variance_selector(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test variance threshold selector."""
        X, _ = sample_data
        selector = FeatureSelector(method="variance", threshold=0.0)

        selected = selector.fit_transform(X)

        # feature4 has zero variance, should be removed
        assert "feature4" not in selected.columns
        assert len(selected.columns) < len(X.columns)

    def test_mutual_info_selector(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test mutual information selector."""
        X, y = sample_data
        selector = FeatureSelector(method="mutual_info", k=3)

        selected = selector.fit_transform(X, y)

        assert len(selected.columns) == 3

    def test_k_best_selector(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test k-best selector."""
        X, y = sample_data
        selector = FeatureSelector(method="k_best", k=2)

        selected = selector.fit_transform(X, y)

        assert len(selected.columns) == 2

    def test_correlation_selector(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test correlation selector."""
        X, _ = sample_data
        selector = FeatureSelector(method="correlation", threshold=0.95)

        selected = selector.fit_transform(X)

        # feature1 and feature2 are highly correlated, one should be removed
        assert len(selected.columns) < len(X.columns)

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid selection method"):
            FeatureSelector(method="invalid")  # type: ignore

    def test_fit_without_target_raises_error(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that methods requiring target raise error without it."""
        X, _ = sample_data
        selector = FeatureSelector(method="mutual_info", k=3)

        with pytest.raises(ValueError, match="Target y required"):
            selector.fit(X)

    def test_get_feature_importance(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test getting feature importance."""
        X, y = sample_data
        selector = FeatureSelector(method="mutual_info", k=3)
        selector.fit(X, y)

        importance = selector.get_feature_importance()

        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(X.columns)

    def test_get_selected_features(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test getting selected features."""
        X, y = sample_data
        selector = FeatureSelector(method="k_best", k=2)
        selector.fit(X, y)

        selected_features = selector.get_selected_features()

        assert len(selected_features) == 2
        assert all(isinstance(f, str) for f in selected_features)

    def test_save_and_load(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test save and load selector."""
        X, y = sample_data
        selector = FeatureSelector(method="k_best", k=2)
        selector.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.joblib"

            # Save
            selector.save(str(path))
            assert path.exists()

            # Load
            loaded_selector = FeatureSelector.load(str(path))
            assert loaded_selector.method == "k_best"
            assert loaded_selector.k == 2

            # Selected features should match
            selected1 = selector.get_selected_features()
            selected2 = loaded_selector.get_selected_features()
            assert selected1 == selected2

    def test_save_unfitted_raises_error(self) -> None:
        """Test that saving unfitted selector raises error."""
        selector = FeatureSelector(method="k_best", k=2)

        with pytest.raises(ValueError, match="Cannot save unfitted"):
            selector.save("selector.joblib")

    def test_load_nonexistent_raises_error(self) -> None:
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureSelector.load("nonexistent.joblib")
