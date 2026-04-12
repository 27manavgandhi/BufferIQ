"""Tests for temporal feature extraction."""


import pandas as pd
import pytest

from bufferiq.ml.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""

    @pytest.fixture
    def extractor(self) -> TemporalFeatureExtractor:
        """Create temporal feature extractor."""
        return TemporalFeatureExtractor()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "published_at": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-02T14:30:00Z",
                    "2024-01-03T18:45:00Z",
                    "2024-01-04T22:15:00Z",
                ],
                "platform": ["linkedin", "twitter", "linkedin", "bluesky"],
            }
        )

    def test_feature_names(self, extractor: TemporalFeatureExtractor) -> None:
        """Test feature names property."""
        names = extractor.feature_names
        assert len(names) == 21
        assert "hour" in names
        assert "day_of_week" in names
        assert "is_weekend" in names

    def test_extract_basic_features(
        self, extractor: TemporalFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test basic feature extraction."""
        features = extractor.extract(sample_df)

        assert len(features) == len(sample_df)
        assert "hour" in features.columns
        assert "day_of_week" in features.columns
        assert "month" in features.columns

        # Check hour extraction
        assert features.iloc[0]["hour"] == 10
        assert features.iloc[1]["hour"] == 14

    def test_weekend_detection(self, extractor: TemporalFeatureExtractor) -> None:
        """Test weekend detection."""
        df = pd.DataFrame(
            {
                "published_at": [
                    "2024-01-06T10:00:00Z",  # Saturday
                    "2024-01-07T10:00:00Z",  # Sunday
                    "2024-01-08T10:00:00Z",  # Monday
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["is_weekend"] == 1
        assert features.iloc[1]["is_weekend"] == 1
        assert features.iloc[2]["is_weekend"] == 0

    def test_business_hours_detection(
        self, extractor: TemporalFeatureExtractor
    ) -> None:
        """Test business hours detection."""
        df = pd.DataFrame(
            {
                "published_at": [
                    "2024-01-01T08:00:00Z",  # Before business hours
                    "2024-01-01T12:00:00Z",  # During business hours
                    "2024-01-01T18:00:00Z",  # After business hours
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["is_business_hours"] == 0
        assert features.iloc[1]["is_business_hours"] == 1
        assert features.iloc[2]["is_business_hours"] == 0

    def test_time_of_day_indicators(self, extractor: TemporalFeatureExtractor) -> None:
        """Test time of day indicators."""
        df = pd.DataFrame(
            {
                "published_at": [
                    "2024-01-01T08:00:00Z",  # Morning
                    "2024-01-01T14:00:00Z",  # Afternoon
                    "2024-01-01T19:00:00Z",  # Evening
                    "2024-01-01T23:00:00Z",  # Night
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["is_morning"] == 1
        assert features.iloc[1]["is_afternoon"] == 1
        assert features.iloc[2]["is_evening"] == 1
        assert features.iloc[3]["is_night"] == 1

    def test_extract_single(self, extractor: TemporalFeatureExtractor) -> None:
        """Test single post extraction."""
        post_data = {
            "published_at": "2024-01-01T10:30:00Z",
            "platform": "linkedin",
        }

        features = extractor.extract_single(post_data)

        assert features["hour"] == 10
        assert features["day_of_week"] == 0  # Monday
        assert features["month"] == 1
        assert features["is_morning"] == 1

    def test_extract_single_empty(self, extractor: TemporalFeatureExtractor) -> None:
        """Test single post extraction with missing data."""
        features = extractor.extract_single({})

        assert all(v == 0 for v in features.values())

    def test_missing_column_raises_error(
        self, extractor: TemporalFeatureExtractor
    ) -> None:
        """Test that missing required column raises error."""
        df = pd.DataFrame({"content": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(df)

    def test_time_since_midnight(self, extractor: TemporalFeatureExtractor) -> None:
        """Test time since midnight calculation."""
        df = pd.DataFrame(
            {"published_at": ["2024-01-01T10:30:00Z", "2024-01-01T14:45:00Z"]}
        )

        features = extractor.extract(df)

        assert features.iloc[0]["time_since_midnight"] == 10 * 60 + 30
        assert features.iloc[1]["time_since_midnight"] == 14 * 60 + 45

    def test_quarter_extraction(self, extractor: TemporalFeatureExtractor) -> None:
        """Test quarter extraction."""
        df = pd.DataFrame(
            {
                "published_at": [
                    "2024-01-01T10:00:00Z",  # Q1
                    "2024-04-01T10:00:00Z",  # Q2
                    "2024-07-01T10:00:00Z",  # Q3
                    "2024-10-01T10:00:00Z",  # Q4
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["quarter"] == 1
        assert features.iloc[1]["quarter"] == 2
        assert features.iloc[2]["quarter"] == 3
        assert features.iloc[3]["quarter"] == 4
