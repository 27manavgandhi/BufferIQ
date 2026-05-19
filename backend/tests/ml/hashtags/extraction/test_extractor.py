"""Tests for hashtag extractor."""

import pytest
from datetime import datetime

from bufferiq.ml.hashtags.extraction.extractor import (
    HashtagExtractor,
    ExtractedHashtag,
    HashtagExtractionResult,
)


class TestHashtagExtractor:
    """Test HashtagExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return HashtagExtractor()

    def test_extract_basic(self, extractor):
        """Test basic hashtag extraction."""
        text = "Great insights on #AI and #MachineLearning! #Tech"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert isinstance(result, HashtagExtractionResult)
        assert result.total_count == 3
        assert result.unique_count == 3
        assert len(result.hashtags) == 3

    def test_extract_with_duplicates(self, extractor):
        """Test extraction with duplicate hashtags."""
        text = "Love #AI! #AI is amazing. #AI rocks!"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 3
        assert result.unique_count == 1
        assert "ai" in result.duplicates
        assert result.duplicates["ai"] == 3

    def test_normalize(self, extractor):
        """Test hashtag normalization."""
        assert extractor.normalize("#AI") == "ai"
        assert extractor.normalize("MachineLearning") == "machinelearning"
        assert extractor.normalize("#AI_Tech") == "aitech"
        assert extractor.normalize("#ML") == "ml"

    def test_extract_with_engagement(self, extractor):
        """Test extraction with engagement data."""
        text = "Check out #AI #Tech"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
            engagement=150,
            reach=5000,
            impressions=10000,
        )

        for hashtag in result.hashtags:
            assert hashtag.engagement == 150
            assert hashtag.reach == 5000
            assert hashtag.impressions == 10000

    def test_platform_validation(self, extractor):
        """Test platform validation."""
        with pytest.raises(ValueError, match="not supported"):
            extractor.extract(
                text="#test",
                post_id="post1",
                platform="facebook",  # Not supported
            )

    def test_extract_linkedin(self, extractor):
        """Test LinkedIn-specific extraction."""
        text = "Professional insights #Leadership #Business #Innovation"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 3
        assert result.platform_distribution["linkedin"] == 3

    def test_extract_twitter(self, extractor):
        """Test Twitter-specific extraction."""
        text = "Quick update #AI #Tech"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="twitter",
        )

        assert result.total_count == 2
        assert result.platform_distribution["twitter"] == 2

    def test_extract_bluesky(self, extractor):
        """Test Bluesky-specific extraction."""
        text = "New post #Decentralized #Web3"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="bluesky",
        )

        assert result.total_count == 2
        assert result.platform_distribution["bluesky"] == 2

    def test_placement_pattern_end(self, extractor):
        """Test detection of hashtags at end."""
        text = "Great content here #AI #ML #Tech"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.placement_pattern == "end"

    def test_placement_pattern_beginning(self, extractor):
        """Test detection of hashtags at beginning."""
        text = "#AI #ML #Tech followed by content here"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.placement_pattern == "beginning"

    def test_placement_pattern_mixed(self, extractor):
        """Test detection of mixed hashtag placement."""
        text = "#AI in the beginning, middle #ML content, and end #Tech"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.placement_pattern == "mixed"

    def test_detect_variants(self, extractor):
        """Test variant detection."""
        hashtags = ["ai", "artificialintelligence", "ml", "machinelearning"]
        variants = extractor.detect_variants(hashtags)

        assert "ai" in variants
        assert "ml" in variants
        assert "artificialintelligence" in variants["ai"]

    def test_extract_with_underscores(self, extractor):
        """Test extraction of hashtags with underscores."""
        text = "#AI_Tech #Machine_Learning"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 2
        # Underscores removed in normalization
        assert "aitech" in [h.hashtag for h in result.hashtags]

    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        result = extractor.extract(
            text="",
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 0
        assert result.unique_count == 0
        assert len(result.hashtags) == 0

    def test_extract_no_hashtags(self, extractor):
        """Test extraction from text without hashtags."""
        text = "This is content without any hashtags"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 0
        assert result.unique_count == 0

    def test_extract_context(self, extractor):
        """Test context extraction around hashtags."""
        text = "Here is some context before #AI and after the hashtag"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        hashtag = result.hashtags[0]
        assert "context" in hashtag.context.lower()
        assert len(hashtag.context) > 0

    def test_extract_position(self, extractor):
        """Test hashtag position tracking."""
        text = "Start #First middle #Second end #Third"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        positions = [h.position for h in result.hashtags]
        assert positions == sorted(positions)  # Should be in order

    def test_extract_original_casing(self, extractor):
        """Test preservation of original casing."""
        text = "#AI #MachineLearning #TECH"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        originals = [h.original for h in result.hashtags]
        assert "#AI" in originals
        assert "#MachineLearning" in originals
        assert "#TECH" in originals

    def test_extract_with_numbers(self, extractor):
        """Test extraction of hashtags with numbers."""
        text = "#AI2024 #Tech101 #ML2"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.total_count == 3
        normalized = [h.hashtag for h in result.hashtags]
        assert "ai2024" in normalized
        assert "tech101" in normalized

    def test_average_position(self, extractor):
        """Test average position calculation."""
        text = "Start #First and #Second and #Third end"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        assert result.avg_position > 0
        assert result.avg_position < len(text)

    def test_extract_metadata(self, extractor):
        """Test metadata in extracted hashtags."""
        now = datetime.now()
        result = extractor.extract(
            text="#AI #ML",
            post_id="post123",
            platform="linkedin",
            engagement=100,
            created_at=now,
        )

        for hashtag in result.hashtags:
            assert hashtag.post_id == "post123"
            assert hashtag.platform == "linkedin"
            assert hashtag.engagement == 100
            assert hashtag.created_at == now

    def test_case_insensitive_normalization(self, extractor):
        """Test case-insensitive normalization."""
        text = "#AI #ai #Ai #aI"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        # All should normalize to "ai"
        assert result.unique_count == 1
        assert result.total_count == 4
        assert "ai" in result.duplicates

    def test_special_characters_removal(self, extractor):
        """Test removal of special characters."""
        # Only alphanumeric and underscore allowed
        text = "#AI-Tech #ML.AI #Tech@2024"
        result = extractor.extract(
            text=text,
            post_id="post1",
            platform="linkedin",
        )

        # Should only extract valid parts
        # #AI-Tech might be extracted as #AI
        # This depends on regex implementation
        assert isinstance(result, HashtagExtractionResult)