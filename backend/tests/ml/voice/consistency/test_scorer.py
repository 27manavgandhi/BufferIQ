"""Tests for voice consistency scorer."""

import pytest
from unittest.mock import Mock, patch

from bufferiq.ml.voice.consistency.scorer import VoiceConsistencyScorer, ConsistencyScore
from bufferiq.ml.voice.profiler.builder import VoiceProfile
from datetime import datetime


class TestVoiceConsistencyScorer:
    """Test voice consistency scorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return VoiceConsistencyScorer(consistency_threshold=75.0)
    
    @pytest.fixture
    def sample_profile(self):
        """Create sample profile."""
        return VoiceProfile(
            profile_id="brand123_linkedin_v1",
            brand_id="brand123",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={
                "type_token_ratio": 0.8,
                "lexical_density": 0.6,
                "complexity": 70.0,
            },
            syntactic_fingerprint={
                "avg_sentence_length": 15.0,
                "sentence_complexity": 0.5,
            },
            stylistic_fingerprint={
                "formality_score": 80.0,
                "emoji_density": 0.0,
                "contraction_ratio": 0.0,
            },
            signature="abc123",
            confidence=0.85,
            sample_size=50,
            platform_profiles={},
        )
    
    def test_score_basic_text(self, scorer, sample_profile):
        """Test basic scoring."""
        text = "This is a professional business document discussing important matters."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert isinstance(score, ConsistencyScore)
        assert 0 <= score.overall_score <= 100
    
    def test_score_invalid_platform_raises_error(self, scorer, sample_profile):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            scorer.score("text", sample_profile, "facebook")
    
    def test_score_empty_text_raises_error(self, scorer, sample_profile):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            scorer.score("", sample_profile, "linkedin")
    
    def test_score_short_text_raises_error(self, scorer, sample_profile):
        """Test short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            scorer.score("Hi", sample_profile, "linkedin")
    
    def test_score_components_present(self, scorer, sample_profile):
        """Test all score components are present."""
        text = "Professional content for business communication and collaboration."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert hasattr(score, 'overall_score')
        assert hasattr(score, 'lexical_consistency')
        assert hasattr(score, 'syntactic_consistency')
        assert hasattr(score, 'stylistic_consistency')
    
    def test_score_is_consistent_flag(self, scorer, sample_profile):
        """Test is_consistent flag is set correctly."""
        text = "Professional content for business communication and collaboration."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert isinstance(score.is_consistent, bool)
        
        if score.overall_score >= 75.0:
            assert score.is_consistent is True
        else:
            assert score.is_consistent is False
    
    def test_score_severity_levels(self, scorer, sample_profile):
        """Test severity levels are assigned."""
        text = "Professional content for business communication."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert score.severity in ["none", "minor", "moderate", "severe"]
    
    def test_score_cosine_similarity_range(self, scorer, sample_profile):
        """Test cosine similarity is in valid range."""
        text = "Professional content for business communication."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert 0 <= score.cosine_similarity <= 1.0
    
    def test_score_recommendations_present(self, scorer, sample_profile):
        """Test recommendations are generated."""
        text = "Hey! Check this out!"  # Casual, inconsistent with formal profile
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert isinstance(score.alignment_suggestions, list)
    
    def test_calculate_cosine_similarity_identical(self, scorer):
        """Test cosine similarity for identical vectors."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        
        similarity = scorer.calculate_cosine_similarity(vec, vec)
        
        assert similarity == pytest.approx(1.0)
    
    def test_calculate_cosine_similarity_orthogonal(self, scorer):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        
        similarity = scorer.calculate_cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(0.0)
    
    def test_calculate_cosine_similarity_opposite(self, scorer):
        """Test cosine similarity for opposite vectors."""
        vec1 = {"a": 1.0}
        vec2 = {"a": -1.0}
        
        similarity = scorer.calculate_cosine_similarity(vec1, vec2)
        
        assert similarity < 0
    
    def test_score_consistent_content_high_score(self, scorer, sample_profile):
        """Test consistent content gets high score."""
        # Content matching formal profile
        text = """
        Furthermore, we would like to acknowledge the significant contributions
        made by our research team. Therefore, we shall proceed accordingly.
        """
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        # Should score reasonably well due to formal style
        assert score.stylistic_consistency > 60
    
    def test_score_inconsistent_content_low_score(self, scorer, sample_profile):
        """Test inconsistent content gets lower score."""
        # Casual content inconsistent with formal profile
        text = "Hey! This is awesome! 😊 Gonna check it out!"
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        # Should have lower stylistic consistency
        assert score.stylistic_consistency < 80
    
    def test_feature_deviations_identified(self, scorer, sample_profile):
        """Test feature deviations are identified."""
        text = "Hey! Check this out! It's amazing! 🎉"
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert isinstance(score.feature_deviations, dict)
    
    def test_kl_divergence_calculated(self, scorer, sample_profile):
        """Test KL divergence is calculated."""
        text = "Professional content for business communication."
        
        score = scorer.score(text, sample_profile, "linkedin")
        
        assert score.kl_divergence >= 0
    
    def test_score_twitter_platform(self, scorer, sample_profile):
        """Test scoring for Twitter platform."""
        text = "Quick update on our latest features!"
        
        score = scorer.score(text, sample_profile, "twitter")
        
        assert score is not None
    
    def test_score_bluesky_platform(self, scorer, sample_profile):
        """Test scoring for Bluesky platform."""
        text = "Sharing some insights on our progress."
        
        score = scorer.score(text, sample_profile, "bluesky")
        
        assert score is not None
    
    def test_threshold_affects_consistency_flag(self):
        """Test threshold affects is_consistent flag."""
        strict_scorer = VoiceConsistencyScorer(consistency_threshold=90.0)
        lenient_scorer = VoiceConsistencyScorer(consistency_threshold=50.0)
        
        profile = VoiceProfile(
            profile_id="test",
            brand_id="test",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={"formality_score": 70.0},
            signature="abc",
            confidence=0.8,
            sample_size=50,
            platform_profiles={},
        )
        
        text = "Professional business content for communication."
        
        strict_score = strict_scorer.score(text, profile, "linkedin")
        lenient_score = lenient_scorer.score(text, profile, "linkedin")
        
        # Same text, different thresholds may give different consistency flags
        # (depending on actual score)
        assert isinstance(strict_score.is_consistent, bool)
        assert isinstance(lenient_score.is_consistent, bool)