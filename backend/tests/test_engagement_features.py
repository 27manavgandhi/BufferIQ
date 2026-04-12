"""Tests for engagement feature extraction."""

import pandas as pd
import pytest

from bufferiq.ml.features.engagement import EngagementFeatureExtractor


class TestEngagementFeatureExtractor:
    """Test engagement feature extraction."""

    @pytest.fixture
    def extractor(self) -> EngagementFeatureExtractor:
        """Create engagement feature extractor."""
        return EngagementFeatureExtractor()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with engagement data."""
        return pd.DataFrame(
            {
                "likes": [10, 20, 15, 30, 25],
                "comments": [2, 4, 3, 6, 5],
                "shares": [1, 2, 1, 3, 2],
                "impressions": [100, 200, 150, 300, 250],
                "user_id": [1, 1, 1, 2, 2],
                "platform": ["linkedin", "linkedin", "twitter", "twitter", "bluesky"],
                "published_at": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-02T10:00:00Z",
                    "2024-01-03T10:00:00Z",
                    "2024-01-04T10:00:00Z",
                    "2024-01-05T10:00:00Z",
                ],
            }
        )

    def test_feature_names(self, extractor: EngagementFeatureExtractor) -> None:
        """Test feature names property."""
        names = extractor.feature_names
        assert len(names) == 15
        assert "user_avg_likes" in names
        assert "platform_avg_engagement_rate" in names
        assert "engagement_trend" in names

    def test_extract_basic_features(
        self, extractor: EngagementFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test basic feature extraction."""
        features = extractor.extract(sample_df)

        assert len(features) == len(sample_df)
        assert "user_avg_likes" in features.columns
        assert "platform_avg_likes" in features.columns

    def test_user_level_features(
        self, extractor: EngagementFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test user-level aggregation features."""
        features = extractor.extract(sample_df)

        # User 1 posts: likes [10, 20, 15]
        user_1_avg = (10 + 20 + 15) / 3
        assert features.iloc[0]["user_avg_likes"] == pytest.approx(user_1_avg, rel=0.01)

        # User 2 posts: likes [30, 25]
        user_2_avg = (30 + 25) / 2
        assert features.iloc[3]["user_avg_likes"] == pytest.approx(user_2_avg, rel=0.01)

    def test_platform_level_features(
        self, extractor: EngagementFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test platform-level aggregation features."""
        features = extractor.extract(sample_df)

        # LinkedIn posts: likes [10, 20]
        linkedin_avg = (10 + 20) / 2
        assert features.iloc[0]["platform_avg_likes"] == pytest.approx(
            linkedin_avg, rel=0.01
        )

    def test_engagement_rate_calculation(
        self, extractor: EngagementFeatureExtractor
    ) -> None:
        """Test engagement rate calculation."""
        df = pd.DataFrame(
            {
                "likes": [10],
                "comments": [2],
                "shares": [1],
                "impressions": [100],
            }
        )

        features = extractor.extract(df)

        # Engagement rate = (10 + 2 + 1) / 100 * 100 = 13%
        expected_rate = 13.0
        assert features.iloc[0]["user_avg_engagement_rate"] == pytest.approx(
            expected_rate, rel=0.01
        )

    def test_missing_columns_returns_zeros(
        self, extractor: EngagementFeatureExtractor
    ) -> None:
        """Test that missing engagement columns returns zero features."""
        df = pd.DataFrame({"content": ["test"]})

        features = extractor.extract(df)

        assert all(features.iloc[0] == 0)

    def test_extract_single(self, extractor: EngagementFeatureExtractor) -> None:
        """Test single post extraction."""
        post_data = {
            "likes": 10,
            "comments": 2,
            "shares": 1,
            "impressions": 100,
        }

        features = extractor.extract_single(post_data)

        # Single post returns zeros for historical features
        assert all(v == 0 for v in features.values())

    def test_zero_impressions_handling(
        self, extractor: EngagementFeatureExtractor
    ) -> None:
        """Test handling of zero impressions."""
        df = pd.DataFrame(
            {
                "likes": [10],
                "comments": [2],
                "shares": [1],
                "impressions": [0],  # Zero impressions
            }
        )

        features = extractor.extract(df)

        # Should not raise division by zero error
        assert "user_avg_engagement_rate" in features.columns


class TestEngagementFeatureExtractorWithoutUserId:
    """Test engagement features without user_id column."""

    @pytest.fixture
    def extractor(self) -> EngagementFeatureExtractor:
        """Create engagement feature extractor."""
        return EngagementFeatureExtractor()

    def test_extract_without_user_id(
        self, extractor: EngagementFeatureExtractor
    ) -> None:
        """Test extraction without user_id column."""
        df = pd.DataFrame(
            {
                "likes": [10, 20, 15],
                "comments": [2, 4, 3],
                "shares": [1, 2, 1],
                "impressions": [100, 200, 150],
            }
        )

        features = extractor.extract(df)

        # Should use overall average instead of user-specific
        overall_avg_likes = (10 + 20 + 15) / 3
        assert all(features["user_avg_likes"] == overall_avg_likes)
