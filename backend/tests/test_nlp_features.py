"""Tests for NLP feature extraction."""

import pandas as pd
import pytest

from bufferiq.ml.features.nlp import NLPFeatureExtractor


class TestNLPFeatureExtractor:
    """Test NLP feature extraction."""

    @pytest.fixture
    def extractor(self) -> NLPFeatureExtractor:
        """Create NLP feature extractor."""
        return NLPFeatureExtractor()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "content": [
                    "This is a great post! I love it.",
                    "This is terrible. I hate it.",
                    "Neutral statement here.",
                    "",
                ]
            }
        )

    def test_feature_names(self, extractor: NLPFeatureExtractor) -> None:
        """Test feature names property."""
        names = extractor.feature_names
        assert len(names) == 15
        assert "sentiment_polarity" in names
        assert "sentiment_subjectivity" in names
        assert "lexical_diversity" in names

    def test_extract_basic_features(
        self, extractor: NLPFeatureExtractor, sample_df: pd.DataFrame
    ) -> None:
        """Test basic feature extraction."""
        features = extractor.extract(sample_df)

        assert len(features) == len(sample_df)
        assert "sentiment_polarity" in features.columns
        assert "lexical_diversity" in features.columns

    def test_sentiment_polarity_range(self, extractor: NLPFeatureExtractor) -> None:
        """Test sentiment polarity is in valid range."""
        df = pd.DataFrame(
            {
                "content": [
                    "Great amazing wonderful!",
                    "Terrible awful horrible!",
                ]
            }
        )

        features = extractor.extract(df)

        # Polarity should be between -1 and 1
        assert -1 <= features.iloc[0]["sentiment_polarity"] <= 1
        assert -1 <= features.iloc[1]["sentiment_polarity"] <= 1

    def test_lexical_diversity(self, extractor: NLPFeatureExtractor) -> None:
        """Test lexical diversity calculation."""
        df = pd.DataFrame(
            {
                "content": [
                    "word word word word",  # Low diversity
                    "every word is different here",  # High diversity
                ]
            }
        )

        features = extractor.extract(df)

        assert 0 <= features.iloc[0]["lexical_diversity"] <= 1
        assert (
            features.iloc[1]["lexical_diversity"]
            > features.iloc[0]["lexical_diversity"]
        )

    def test_stopword_ratio(self, extractor: NLPFeatureExtractor) -> None:
        """Test stopword ratio calculation."""
        df = pd.DataFrame(
            {
                "content": [
                    "the and a the and",  # High stopword ratio
                    "python programming language",  # Low stopword ratio
                ]
            }
        )

        features = extractor.extract(df)

        assert 0 <= features.iloc[0]["stopword_ratio"] <= 1
        assert features.iloc[0]["stopword_ratio"] > features.iloc[1]["stopword_ratio"]

    def test_extract_single(self, extractor: NLPFeatureExtractor) -> None:
        """Test single post extraction."""
        post_data = {"content": "This is a great post!"}

        features = extractor.extract_single(post_data)

        assert "sentiment_polarity" in features
        assert "lexical_diversity" in features
        assert -1 <= features["sentiment_polarity"] <= 1

    def test_extract_single_empty(self, extractor: NLPFeatureExtractor) -> None:
        """Test single post extraction with empty content."""
        features = extractor.extract_single({})

        assert all(v == 0 for v in features.values())

    def test_missing_content_column(self, extractor: NLPFeatureExtractor) -> None:
        """Test that missing content column raises error."""
        df = pd.DataFrame({"other": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(df)

    def test_null_content_handling(self, extractor: NLPFeatureExtractor) -> None:
        """Test handling of null content."""
        df = pd.DataFrame({"content": [None, "", "test"]})

        features = extractor.extract(df)

        assert features.iloc[0]["sentiment_polarity"] == 0
        assert features.iloc[1]["sentiment_polarity"] == 0
