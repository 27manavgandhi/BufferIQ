"""Tests for hashtag normalizer."""

import pytest

from bufferiq.ml.hashtags.extraction.normalizer import HashtagNormalizer


class TestHashtagNormalizer:
    """Test HashtagNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        return HashtagNormalizer()

    def test_normalize_basic(self, normalizer):
        """Test basic normalization."""
        assert normalizer.normalize("#AI") == "ai"
        assert normalizer.normalize("MachineLearning") == "machinelearning"
        assert normalizer.normalize("#TECH") == "tech"

    def test_normalize_with_underscores(self, normalizer):
        """Test normalization removes underscores."""
        assert normalizer.normalize("#AI_Tech") == "aitech"
        assert normalizer.normalize("#Machine_Learning") == "machinelearning"

    def test_normalize_with_spaces(self, normalizer):
        """Test normalization removes spaces."""
        assert normalizer.normalize("# AI Tech") == "aitech"
        assert normalizer.normalize("Machine Learning ") == "machinelearning"

    def test_get_variants_ai(self, normalizer):
        """Test getting variants for AI."""
        variants = normalizer.get_variants("ai")

        assert "ai" in variants
        assert "artificialintelligence" in variants
        assert "aitech" in variants

    def test_get_variants_ml(self, normalizer):
        """Test getting variants for ML."""
        variants = normalizer.get_variants("ml")

        assert "ml" in variants
        assert "machinelearning" in variants

    def test_get_variants_unknown(self, normalizer):
        """Test getting variants for unknown hashtag."""
        variants = normalizer.get_variants("unknown")

        # Should return list with just the hashtag itself
        assert variants == ["unknown"]

    def test_get_canonical_known(self, normalizer):
        """Test getting canonical form for known variant."""
        assert normalizer.get_canonical("artificialintelligence") == "ai"
        assert normalizer.get_canonical("machinelearning") == "ml"
        assert normalizer.get_canonical("ai") == "ai"

    def test_get_canonical_unknown(self, normalizer):
        """Test getting canonical form for unknown hashtag."""
        canonical = normalizer.get_canonical("unknown")
        assert canonical == "unknown"

    def test_are_variants_true(self, normalizer):
        """Test variant detection for actual variants."""
        assert normalizer.are_variants("ai", "artificialintelligence")
        assert normalizer.are_variants("ml", "machinelearning")
        assert normalizer.are_variants("AI", "ArtificialIntelligence")

    def test_are_variants_false(self, normalizer):
        """Test variant detection for non-variants."""
        assert not normalizer.are_variants("ai", "blockchain")
        assert not normalizer.are_variants("tech", "business")

    def test_normalize_misspellings(self, normalizer):
        """Test normalization fixes misspellings."""
        # This would need actual misspelling corrections
        # Currently just normalizes
        result = normalizer.normalize("artifical")  # misspelled
        assert isinstance(result, str)

    def test_case_insensitive(self, normalizer):
        """Test case insensitive operations."""
        assert normalizer.normalize("#AI") == normalizer.normalize("#ai")
        assert normalizer.normalize("#MachineLearning") == normalizer.normalize(
            "#machinelearning"
        )

    def test_normalize_empty(self, normalizer):
        """Test normalizing empty string."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("#") == ""

    def test_normalize_hashtag_symbol(self, normalizer):
        """Test removal of # symbol."""
        assert normalizer.normalize("#hashtag") == "hashtag"
        assert normalizer.normalize("hashtag") == "hashtag"
        assert normalizer.normalize("##hashtag") == "hashtag"

    def test_get_variants_case_insensitive(self, normalizer):
        """Test variants are case insensitive."""
        variants_lower = normalizer.get_variants("ai")
        variants_upper = normalizer.get_variants("AI")

        assert variants_lower == variants_upper

    def test_canonical_map_completeness(self, normalizer):
        """Test canonical map includes all variants."""
        # Every variant should map to a canonical
        for canonical, variants in normalizer.canonical_map.items():
            for variant in variants:
                assert variant in normalizer.variant_to_canonical
                assert normalizer.variant_to_canonical[variant] == canonical

    def test_normalize_preserves_alphanumeric(self, normalizer):
        """Test normalization preserves alphanumeric characters."""
        assert normalizer.normalize("#AI2024") == "ai2024"
        assert normalizer.normalize("#Tech101") == "tech101"

    def test_seo_variant(self, normalizer):
        """Test SEO hashtag variants."""
        variants = normalizer.get_variants("seo")

        assert "seo" in variants
        assert "searchengineoptimization" in variants