"""Tests for platform-specific feature extraction."""

import pandas as pd
import pytest

from bufferiq.ml.features.base import validate_platform
from bufferiq.ml.features.platform_specific import PlatformSpecificFeatureExtractor


class TestPlatformValidation:
    """Test platform validation."""

    def test_validate_supported_platforms(self) -> None:
        """Test that supported platforms are accepted."""
        for platform in ["linkedin", "twitter", "bluesky"]:
            validate_platform(platform)  # Should not raise

    def test_validate_rejects_facebook(self) -> None:
        """Test that facebook is rejected."""
        with pytest.raises(ValueError, match="facebook.*not supported"):
            validate_platform("facebook")

    def test_validate_rejects_invalid_platform(self) -> None:
        """Test that invalid platforms are rejected."""
        with pytest.raises(ValueError, match="not supported"):
            validate_platform("invalid")


class TestPlatformSpecificFeatureExtractor:
    """Test platform-specific feature extraction."""

    @pytest.fixture
    def extractor(self) -> PlatformSpecificFeatureExtractor:
        """Create platform-specific feature extractor."""
        return PlatformSpecificFeatureExtractor()

    @pytest.fixture
    def linkedin_df(self) -> pd.DataFrame:
        """Create LinkedIn sample DataFrame."""
        return pd.DataFrame(
            {
                "content": [
                    "Excited to announce our new hiring initiative! Join our team. #leadership #hiring",
                    "Simple post",
                ],
                "platform": ["linkedin", "linkedin"],
            }
        )

    @pytest.fixture
    def twitter_df(self) -> pd.DataFrame:
        """Create Twitter sample DataFrame."""
        return pd.DataFrame(
            {
                "content": [
                    "@user check this out! #tech",
                    "Thread: 🧵 First tweet here",
                ],
                "platform": ["twitter", "twitter"],
            }
        )

    @pytest.fixture
    def bluesky_df(self) -> pd.DataFrame:
        """Create Bluesky sample DataFrame."""
        return pd.DataFrame(
            {
                "content": [
                    "The AT Protocol enables true decentralization",
                    "Simple post",
                ],
                "platform": ["bluesky", "bluesky"],
            }
        )

    def test_feature_names(self, extractor: PlatformSpecificFeatureExtractor) -> None:
        """Test feature names property."""
        names = extractor.feature_names
        assert len(names) == 16
        assert "is_professional_tone" in names
        assert "is_thread_starter" in names
        assert "is_decentralization_topic" in names

    def test_extract_linkedin_features(
        self, extractor: PlatformSpecificFeatureExtractor, linkedin_df: pd.DataFrame
    ) -> None:
        """Test LinkedIn feature extraction."""
        features = extractor.extract(linkedin_df)

        # First post has career keywords and industry hashtags
        assert features.iloc[0]["has_career_keywords"] == 1
        assert features.iloc[0]["has_industry_hashtags"] == 1

        # LinkedIn features should be set, others should be 0
        assert features.iloc[0]["is_thread_starter"] == 0
        assert features.iloc[0]["is_decentralization_topic"] == 0

    def test_extract_twitter_features(
        self, extractor: PlatformSpecificFeatureExtractor, twitter_df: pd.DataFrame
    ) -> None:
        """Test Twitter feature extraction."""
        features = extractor.extract(twitter_df)

        # First post is a reply (starts with @)
        assert features.iloc[0]["is_reply"] == 1

        # Second post is a thread starter (has 🧵)
        assert features.iloc[1]["is_thread_starter"] == 1

        # Twitter features should be set, others should be 0
        assert features.iloc[0]["is_professional_tone"] == 0
        assert features.iloc[0]["is_decentralization_topic"] == 0

    def test_extract_bluesky_features(
        self, extractor: PlatformSpecificFeatureExtractor, bluesky_df: pd.DataFrame
    ) -> None:
        """Test Bluesky feature extraction."""
        features = extractor.extract(bluesky_df)

        # First post has decentralization keywords
        assert features.iloc[0]["is_decentralization_topic"] == 1

        # Bluesky features should be set, others should be 0
        assert features.iloc[0]["is_professional_tone"] == 0
        assert features.iloc[0]["is_thread_starter"] == 0

    def test_extract_single_linkedin(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test single post extraction for LinkedIn."""
        post_data = {
            "content": "Excited to announce we're hiring! #leadership",
            "platform": "linkedin",
        }

        features = extractor.extract_single(post_data)

        assert features["has_career_keywords"] == 1
        assert features["is_thread_starter"] == 0

    def test_extract_single_twitter(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test single post extraction for Twitter."""
        post_data = {
            "content": "@user great post!",
            "platform": "twitter",
        }

        features = extractor.extract_single(post_data)

        assert features["is_reply"] == 1
        assert features["is_professional_tone"] == 0

    def test_extract_single_bluesky(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test single post extraction for Bluesky."""
        post_data = {
            "content": "AT Protocol is amazing",
            "platform": "bluesky",
        }

        features = extractor.extract_single(post_data)

        assert features["is_decentralization_topic"] == 1
        assert features["is_professional_tone"] == 0

    def test_invalid_platform_raises_error(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test that invalid platform raises error."""
        df = pd.DataFrame(
            {
                "content": ["test"],
                "platform": ["facebook"],
            }
        )

        with pytest.raises(ValueError, match="facebook.*not supported"):
            extractor.extract(df)

    def test_extract_single_invalid_platform(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test single post with invalid platform raises error."""
        post_data = {
            "content": "test",
            "platform": "facebook",
        }

        with pytest.raises(ValueError, match="facebook.*not supported"):
            extractor.extract_single(post_data)

    def test_missing_columns_raises_error(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test that missing required columns raises error."""
        df = pd.DataFrame({"content": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(df)

    def test_optimal_length_linkedin(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test optimal length detection for LinkedIn."""
        # Optimal: 1300-1500 chars
        content_optimal = "a" * 1400
        content_short = "a" * 100

        post_optimal = {"content": content_optimal, "platform": "linkedin"}
        post_short = {"content": content_short, "platform": "linkedin"}

        features_optimal = extractor.extract_single(post_optimal)
        features_short = extractor.extract_single(post_short)

        assert features_optimal["optimal_length_linkedin"] == 1
        assert features_short["optimal_length_linkedin"] == 0

    def test_optimal_length_twitter(
        self, extractor: PlatformSpecificFeatureExtractor
    ) -> None:
        """Test optimal length detection for Twitter."""
        # Optimal: 71-100 chars
        content_optimal = "a" * 80
        content_short = "a" * 20

        post_optimal = {"content": content_optimal, "platform": "twitter"}
        post_short = {"content": content_short, "platform": "twitter"}

        features_optimal = extractor.extract_single(post_optimal)
        features_short = extractor.extract_single(post_short)

        assert features_optimal["optimal_length_twitter"] == 1
        assert features_short["optimal_length_twitter"] == 0
