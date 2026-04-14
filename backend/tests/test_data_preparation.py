"""Tests for data preparation."""

import pandas as pd
import pytest

from bufferiq.ml.training.data_preparation import DataPreparation


class TestDataPreparation:
    """Test data preparation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "feature1": list(range(200)),
                "feature2": list(range(200, 400)),
                "target": list(range(100)) * 2,
                "published_at": pd.date_range("2024-01-01", periods=200),
                "platform": ["linkedin"] * 100 + ["twitter"] * 100,
            }
        )

    def test_init(self) -> None:
        """Test initialization."""
        prep = DataPreparation(test_size=0.2, validation_size=0.1)
        assert prep.test_size == 0.2
        assert prep.validation_size == 0.1
        assert prep.random_state == 42

    def test_init_invalid_test_size(self) -> None:
        """Test invalid test size raises error."""
        with pytest.raises(ValueError, match="test_size must be between"):
            DataPreparation(test_size=1.5)

    def test_split_data_basic(self, sample_df: pd.DataFrame) -> None:
        """Test basic data splitting."""
        prep = DataPreparation(test_size=0.2, validation_size=0.1, time_based_split=False)

        X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(
            sample_df, "target", ["feature1", "feature2"]
        )

        # Check sizes
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(sample_df)
        assert len(X_test) == pytest.approx(len(sample_df) * 0.2, abs=5)

    def test_split_data_time_based(self, sample_df: pd.DataFrame) -> None:
        """Test time-based splitting."""
        prep = DataPreparation(test_size=0.2, validation_size=0.1, time_based_split=True)

        X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(
            sample_df, "target", ["feature1", "feature2"], time_column="published_at"
        )

        # Check that test data is most recent
        assert len(X_test) > 0
        assert len(X_train) > len(X_test)

    def test_split_data_missing_target(self, sample_df: pd.DataFrame) -> None:
        """Test split with missing target raises error."""
        prep = DataPreparation()

        with pytest.raises(ValueError, match="Target column.*not found"):
            prep.split_data(sample_df, "nonexistent", ["feature1"])

    def test_split_data_missing_features(self, sample_df: pd.DataFrame) -> None:
        """Test split with missing features raises error."""
        prep = DataPreparation()

        with pytest.raises(ValueError, match="Missing feature columns"):
            prep.split_data(sample_df, "target", ["feature1", "nonexistent"])

    def test_split_data_insufficient_samples(self) -> None:
        """Test split with too few samples raises error."""
        prep = DataPreparation()
        small_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [1, 2, 3],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            prep.split_data(small_df, "target", ["feature1"])

    def test_validate_data_quality(self, sample_df: pd.DataFrame) -> None:
        """Test data quality validation."""
        prep = DataPreparation()
        quality = prep.validate_data_quality(sample_df)

        assert "total_samples" in quality
        assert quality["total_samples"] == len(sample_df)
        assert "missing_values" in quality
        assert "warnings" in quality

    def test_validate_data_quality_with_missing(self) -> None:
        """Test quality validation with missing values."""
        prep = DataPreparation()
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4],
                "feature2": [1, None, 3, 4],
            }
        )

        quality = prep.validate_data_quality(df)

        assert len(quality["missing_values"]) > 0
        assert any("Missing values" in w for w in quality["warnings"])

    def test_handle_missing_values_drop(self) -> None:
        """Test dropping missing values."""
        prep = DataPreparation()
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4],
                "feature2": [1, 2, 3, 4],
            }
        )

        df_clean = prep.handle_missing_values(df, strategy="drop")

        assert len(df_clean) == 3
        assert not df_clean.isnull().any().any()

    def test_handle_missing_values_median(self) -> None:
        """Test filling missing values with median."""
        prep = DataPreparation()
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0],
            }
        )

        df_clean = prep.handle_missing_values(df, strategy="median")

        assert not df_clean.isnull().any().any()
        assert df_clean["feature1"].iloc[2] == 2.0

    def test_remove_outliers_iqr(self) -> None:
        """Test outlier removal with IQR method."""
        prep = DataPreparation()
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # 100 is outlier
            }
        )

        df_clean = prep.remove_outliers(df, ["feature1"], method="iqr")

        assert len(df_clean) < len(df)
        assert df_clean["feature1"].max() < 100