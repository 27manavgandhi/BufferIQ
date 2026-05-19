"""Tests for strategy generator."""

import pytest

from bufferiq.ml.hashtags.strategy.generator import (
    HashtagStrategyGenerator,
    HashtagStrategy,
)


class TestHashtagStrategyGenerator:
    """Test HashtagStrategyGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return HashtagStrategyGenerator()

    def test_generate_linkedin(self, generator):
        """Test LinkedIn strategy generation."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="artificial intelligence",
        )

        assert isinstance(strategy, HashtagStrategy)
        assert strategy.platform == "linkedin"
        assert 3 <= strategy.recommended_count <= 5

    def test_generate_twitter(self, generator):
        """Test Twitter strategy generation."""
        strategy = generator.generate(
            platform="twitter",
            content_topic="artificial intelligence",
        )

        assert strategy.platform == "twitter"
        assert 1 <= strategy.recommended_count <= 2

    def test_generate_bluesky(self, generator):
        """Test Bluesky strategy generation."""
        strategy = generator.generate(
            platform="bluesky",
            content_topic="artificial intelligence",
        )

        assert strategy.platform == "bluesky"
        assert 1 <= strategy.recommended_count <= 3

    def test_generate_invalid_platform(self, generator):
        """Test with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            generator.generate(
                platform="facebook",
                content_topic="test",
            )

    def test_strategy_has_hashtags(self, generator):
        """Test strategy includes hashtags."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="artificial intelligence",
        )

        assert len(strategy.recommended_hashtags) > 0
        assert len(strategy.recommended_hashtags) == strategy.recommended_count

    def test_strategy_mix_breakdown(self, generator):
        """Test mix breakdown."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="artificial intelligence",
        )

        assert isinstance(strategy.broad_hashtags, list)
        assert isinstance(strategy.niche_hashtags, list)
        assert isinstance(strategy.branded_hashtags, list)

    def test_strategy_placement(self, generator):
        """Test placement recommendation."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="test",
        )

        assert strategy.placement in ["beginning", "end", "first_comment"]

    def test_strategy_predicted_engagement(self, generator):
        """Test predicted engagement."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="artificial intelligence",
        )

        assert strategy.predicted_engagement > 0
        assert isinstance(strategy.predicted_engagement, float)

    def test_linkedin_optimal_count(self, generator):
        """Test LinkedIn optimal count is 5."""
        strategy = generator.generate(
            platform="linkedin",
            content_topic="test",
        )

        # LinkedIn optimal is 5
        assert strategy.recommended_count <= 5

    def test_twitter_optimal_count(self, generator):
        """Test Twitter optimal count is 2."""
        strategy = generator.generate(
            platform="twitter",
            content_topic="test",
        )

        # Twitter optimal is 2
        assert strategy.recommended_count <= 2


class TestHashtagMixer:
    """Test HashtagMixer class."""

    @pytest.fixture
    def mixer(self):
        """Create mixer instance."""
        from bufferiq.ml.hashtags.strategy.mixer import HashtagMixer
        return HashtagMixer()

    def test_create_mix(self, mixer):
        """Test creating hashtag mix."""
        broad = ["ai", "tech", "innovation"]
        niche = ["aitips", "mlbasics"]
        branded = ["mycompany"]

        mix = mixer.create_mix(
            broad=broad,
            niche=niche,
            branded=branded,
            total_count=5,
            platform="linkedin",
        )

        assert len(mix) == 5
        assert all(isinstance(h, str) for h in mix)

    def test_create_mix_respects_total(self, mixer):
        """Test mix respects total count."""
        broad = ["ai", "tech", "innovation", "digital", "future"]
        niche = ["aitips", "mlbasics", "techtrends"]
        branded = ["brand"]

        mix = mixer.create_mix(
            broad=broad,
            niche=niche,
            branded=branded,
            total_count=3,
            platform="linkedin",
        )

        assert len(mix) == 3

    def test_calculate_diversity_score(self, mixer):
        """Test diversity score calculation."""
        hashtags = ["ai", "machinelearning", "tech", "innovation"]

        diversity = mixer.calculate_diversity_score(hashtags)

        assert 0 <= diversity <= 100

    def test_diversity_empty_list(self, mixer):
        """Test diversity with empty list."""
        diversity = mixer.calculate_diversity_score([])

        assert diversity == 0.0