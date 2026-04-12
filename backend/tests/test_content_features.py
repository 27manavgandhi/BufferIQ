"""Tests for content feature extraction."""

import pandas as pd
import pytest

from bufferiq.ml.features.content import ContentFeatureExtractor


class TestContentFeatureExtractor:
    """Test content feature extraction."""

    @pytest.fixture
    def extractor(self) -> ContentFeatureExtractor:
        """Create content feature extractor."""
        return ContentFeatureExtractor()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "content": [
                    "Hello world! Check out https://example.com #test @user",
                    "Simple post",
                    "Post with emoji 😀 and numbers 123",
                    "",
                ]
            }
        )

    def test_feature_names(self, extractor: ContentFeatureExtractor) -> None:
        """Test feature names property."""
        names = extractor.feature_names
        assert len(names) == 25
        assert "text_length" in names
        assert "word_count" in names
        assert "hashtag_count" in names

    def test_extract_basic_features(
        self, extractor: ContentFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test basic feature extraction."""
        features = extractor.extract(sample_df)

        assert len(features) == len(sample_df)
        assert features.iloc[0]["text_length"] > 0
        assert features.iloc[0]["word_count"] > 0

    def test_url_detection(self, extractor: ContentFeatureExtractor) -> None:
        """Test URL detection."""
        df = pd.DataFrame(
            {
                "content": [
                    "Check out https://example.com",
                    "No URL here",
                    "Multiple URLs: https://a.com and https://b.com",
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["has_url"] == 1
        assert features.iloc[0]["url_count"] == 1
        assert features.iloc[1]["has_url"] == 0
        assert features.iloc[2]["url_count"] == 2

    def test_hashtag_detection(self, extractor: ContentFeatureExtractor) -> None:
        """Test hashtag detection."""
        df = pd.DataFrame(
            {
                "content": [
                    "Post with #hashtag",
                    "No hashtag",
                    "Multiple #hashtags #here",
                ]
            }
        )

        features = extractor.extract(df)

        assert features.iloc[0]["has_hashtag"] == 1
        assert features.iloc[0]["hashtag_count"] == 1
        assert features.iloc[1]["has_hashtag"] == 0
        assert features.iloc[2]["hashtag_count"] == 2

    def test_mention_detection(self, extractor: ContentFeatureExtractor) -> None:
        """Test mention detection."""
        df = pd.DataFrame({"content": ["Hey @user", "No mention", "@user1 @user2"]})

        features = extractor.extract(df)

        assert features.iloc[0]["has_mention"] == 1
        assert features.iloc[0]["mention_count"] == 1
        assert features.iloc[2]["mention_count"] == 2

    def test_emoji_detection(self, extractor: ContentFeatureExtractor) -> None:
        """Test emoji detection."""
        df = pd.DataFrame({"content": ["Happy 😀", "No emoji", "Multiple 😀😁😂"]})

        features = extractor.extract(df)

        assert features.iloc[0]["has_emoji"] == 1
        assert features.iloc[0]["emoji_count"] >= 1
        assert features.iloc[1]["has_emoji"] == 0

    def test_question_exclamation_detection(
        self, extractor: ContentFeatureExtractor
    ) -> None:
        """Test question and exclamation detection."""
        df = pd.DataFrame(
            {"content": ["Is this a question?", "Exciting!", "Normal post"]}
        )

        features = extractor.extract(df)

        assert features.iloc[0]["has_question"] == 1
        assert features.iloc[0]["question_count"] == 1
        assert features.iloc[1]["has_exclamation"] == 1
        assert features.iloc[1]["exclamation_count"] == 1

    def test_extract_single(self, extractor: ContentFeatureExtractor) -> None:
        """Test single post extraction."""
        post_data = {"content": "Hello #world! Check https://example.com @user"}

        features = extractor.extract_single(post_data)

        assert features["text_length"] > 0
        assert features["word_count"] > 0
        assert features["has_hashtag"] == 1
        assert features["has_url"] == 1
        assert features["has_mention"] == 1
        assert features["has_exclamation"] == 1

    def test_extract_single_empty(self, extractor: ContentFeatureExtractor) -> None:
        """Test single post extraction with empty content."""
        features = extractor.extract_single({})

        assert all(v == 0 for v in features.values())

    def test_missing_content_column(self, extractor: ContentFeatureExtractor) -> None:
        """Test that missing content column raises error."""
        df = pd.DataFrame({"other": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(df)

    def test_uppercase_ratio(self, extractor: ContentFeatureExtractor) -> None:
        """Test uppercase ratio calculation."""
        df = pd.DataFrame({"content": ["HELLO", "hello", "HeLLo"]})

        features = extractor.extract(df)

        assert features.iloc[0]["uppercase_ratio"] == 1.0
        assert features.iloc[1]["uppercase_ratio"] == 0.0
        assert 0 < features.iloc[2]["uppercase_ratio"] < 1

    def test_null_content_handling(self, extractor: ContentFeatureExtractor) -> None:
        """Test handling of null content."""
        df = pd.DataFrame({"content": [None, "", "test"]})

        features = extractor.extract(df)

        assert features.iloc[0]["text_length"] == 0
        assert features.iloc[1]["text_length"] == 0
        assert features.iloc[2]["text_length"] > 0
