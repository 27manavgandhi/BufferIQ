"""Tests for feature scaler."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bufferiq.ml.features.scaler import FeatureScaler


class TestFeatureScaler:
    """Test feature scaler."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    def test_standard_scaler(self, sample_df: pd.DataFrame) -> None:
        """Test standard scaler."""
        scaler = FeatureScaler(method="standard")
        feature_cols = ["feature1", "feature2", "feature3"]

        scaled = scaler.fit_transform(sample_df, feature_cols)

        # Mean should be close to 0, std close to 1
        assert scaled[feature_cols].mean().abs().max() < 0.01
        assert (scaled[feature_cols].std() - 1.0).abs().max() < 0.01

    def test_minmax_scaler(self, sample_df: pd.DataFrame) -> None:
        """Test minmax scaler."""
        scaler = FeatureScaler(method="minmax")
        feature_cols = ["feature1", "feature2", "feature3"]

        scaled = scaler.fit_transform(sample_df, feature_cols)

        # Values should be between 0 and 1
        assert scaled[feature_cols].min().min() == pytest.approx(0.0)
        assert scaled[feature_cols].max().max() == pytest.approx(1.0)

    def test_robust_scaler(self, sample_df: pd.DataFrame) -> None:
        """Test robust scaler."""
        scaler = FeatureScaler(method="robust")
        feature_cols = ["feature1", "feature2", "feature3"]

        scaled = scaler.fit_transform(sample_df, feature_cols)

        # Median should be close to 0
        assert scaled[feature_cols].median().abs().max() < 0.01

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid scaling method"):
            FeatureScaler(method="invalid")  # type: ignore

    def test_fit_transform_separately(self, sample_df: pd.DataFrame) -> None:
        """Test fit and transform separately."""
        scaler = FeatureScaler(method="standard")
        feature_cols = ["feature1", "feature2", "feature3"]

        scaler.fit(sample_df, feature_cols)
        scaled = scaler.transform(sample_df, feature_cols)

        assert scaled[feature_cols].mean().abs().max() < 0.01

    def test_transform_before_fit_raises_error(self, sample_df: pd.DataFrame) -> None:
        """Test that transform before fit raises error."""
        scaler = FeatureScaler(method="standard")

        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(sample_df, ["feature1"])

    def test_inverse_transform(self, sample_df: pd.DataFrame) -> None:
        """Test inverse transform."""
        scaler = FeatureScaler(method="standard")
        feature_cols = ["feature1", "feature2", "feature3"]

        scaled = scaler.fit_transform(sample_df, feature_cols)
        inversed = scaler.inverse_transform(scaled, feature_cols)

        # Should match original
        pd.testing.assert_frame_equal(
            inversed[feature_cols], sample_df[feature_cols], rtol=0.01
        )

    def test_save_and_load(self, sample_df: pd.DataFrame) -> None:
        """Test save and load scaler."""
        scaler = FeatureScaler(method="standard")
        feature_cols = ["feature1", "feature2", "feature3"]
        scaler.fit(sample_df, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaler.joblib"

            # Save
            scaler.save(str(path))
            assert path.exists()

            # Load
            loaded_scaler = FeatureScaler.load(str(path))
            assert loaded_scaler.method == "standard"
            assert loaded_scaler._feature_columns == feature_cols

            # Transformed data should match
            scaled1 = scaler.transform(sample_df, feature_cols)
            scaled2 = loaded_scaler.transform(sample_df, feature_cols)

            pd.testing.assert_frame_equal(scaled1, scaled2)

    def test_save_unfitted_raises_error(self) -> None:
        """Test that saving unfitted scaler raises error."""
        scaler = FeatureScaler(method="standard")

        with pytest.raises(ValueError, match="Cannot save unfitted"):
            scaler.save("scaler.joblib")

    def test_load_nonexistent_raises_error(self) -> None:
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureScaler.load("nonexistent.joblib")

    def test_missing_columns_raises_error(self, sample_df: pd.DataFrame) -> None:
        """Test that missing columns raises error."""
        scaler = FeatureScaler(method="standard")

        with pytest.raises(ValueError, match="Missing columns"):
            scaler.fit(sample_df, ["nonexistent"])
