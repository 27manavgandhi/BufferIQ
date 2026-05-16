"""Tests for voice profile builder."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from bufferiq.ml.voice.profiler.builder import VoiceProfileBuilder, VoiceProfile
from bufferiq.ml.voice.extraction.extractor import VoiceFeatures
from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalMetrics
from bufferiq.ml.voice.linguistic.syntactic_analyzer import SyntacticMetrics
from bufferiq.ml.voice.stylistic.style_detector import StylisticFeatures, WritingStyle


class TestVoiceProfileBuilder:
    """Test voice profile builder."""
    
    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        return VoiceProfileBuilder()
    
    @pytest.fixture
    def sample_voice_features(self):
        """Create sample voice features."""
        return VoiceFeatures(
            lexical_profile=LexicalMetrics(
                type_token_ratio=0.8,
                hapax_legomena_ratio=0.3,
                average_word_length=5.0,
                vocabulary_size=100,
                unique_words=80,
                word_frequency_dist={"word1": 10, "word2": 8},
                lexical_density=0.6,
                complexity_score=70.0,
            ),
            syntactic_profile=SyntacticMetrics(
                average_sentence_length=15.0,
                sentence_complexity=0.5,
                pos_distribution={"NOUN": 0.3, "VERB": 0.2},
                dependency_depth=2.0,
                clause_density=1.5,
                syntactic_variety=0.6,
            ),
            stylistic_profile=StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.8,
                formality_score=80.0,
                punctuation_density={"period": 5.0},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.1,
                exclamation_ratio=0.0,
                average_paragraph_length=3.0,
            ),
            temporal_evolution={},
            platform_variations={},
            confidence_score=0.85,
            sample_size=50,
            extraction_date=datetime.utcnow(),
        )
    
    def test_build_basic_profile(self, builder, sample_voice_features):
        """Test basic profile building."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert isinstance(profile, VoiceProfile)
        assert profile.brand_id == "brand123"
        assert profile.version == 1
        assert profile.confidence == 0.85
    
    def test_build_invalid_platform_raises_error(self, builder, sample_voice_features):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            builder.build(
                brand_id="brand123",
                voice_features=sample_voice_features,
                platform="facebook",
            )
    
    def test_build_insufficient_sample_raises_error(self, builder):
        """Test insufficient sample size raises error."""
        features = VoiceFeatures(
            lexical_profile=Mock(),
            syntactic_profile=Mock(),
            stylistic_profile=Mock(),
            temporal_evolution={},
            platform_variations={},
            confidence_score=0.5,
            sample_size=5,  # Too small
            extraction_date=datetime.utcnow(),
        )
        
        with pytest.raises(ValueError, match="Insufficient sample"):
            builder.build(
                brand_id="brand123",
                voice_features=features,
                platform="linkedin",
            )
    
    def test_build_creates_fingerprints(self, builder, sample_voice_features):
        """Test profile creates all fingerprints."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert len(profile.lexical_fingerprint) > 0
        assert len(profile.syntactic_fingerprint) > 0
        assert len(profile.stylistic_fingerprint) > 0
    
    def test_build_generates_signature(self, builder, sample_voice_features):
        """Test profile generates signature."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert isinstance(profile.signature, str)
        assert len(profile.signature) == 64  # SHA-256 hex
    
    def test_build_with_previous_version(self, builder, sample_voice_features):
        """Test building with previous version."""
        # Build first version
        profile_v1 = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        # Build second version
        profile_v2 = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
            previous_profile=profile_v1,
        )
        
        assert profile_v2.version == 2
        assert profile_v2.previous_version == profile_v1.profile_id
        assert profile_v2.drift_from_previous is not None
    
    def test_build_profile_id_format(self, builder, sample_voice_features):
        """Test profile ID format."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        # Should be: brand_platform_vN_YYYYMMDD
        assert "brand123" in profile.profile_id
        assert "linkedin" in profile.profile_id
        assert "v1" in profile.profile_id
    
    def test_build_created_at_set(self, builder, sample_voice_features):
        """Test created_at is set."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert isinstance(profile.created_at, datetime)
    
    def test_build_twitter_platform(self, builder, sample_voice_features):
        """Test building for Twitter."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="twitter",
        )
        
        assert "twitter" in profile.profile_id
    
    def test_build_bluesky_platform(self, builder, sample_voice_features):
        """Test building for Bluesky."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="bluesky",
        )
        
        assert "bluesky" in profile.profile_id
    
    def test_lexical_fingerprint_contents(self, builder, sample_voice_features):
        """Test lexical fingerprint contains expected fields."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert "type_token_ratio" in profile.lexical_fingerprint
        assert "lexical_density" in profile.lexical_fingerprint
        assert "complexity" in profile.lexical_fingerprint
    
    def test_syntactic_fingerprint_contents(self, builder, sample_voice_features):
        """Test syntactic fingerprint contains expected fields."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert "avg_sentence_length" in profile.syntactic_fingerprint
        assert "sentence_complexity" in profile.syntactic_fingerprint
    
    def test_stylistic_fingerprint_contents(self, builder, sample_voice_features):
        """Test stylistic fingerprint contains expected fields."""
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert "formality_score" in profile.stylistic_fingerprint
        assert "emoji_density" in profile.stylistic_fingerprint
    
    def test_drift_calculation(self, builder, sample_voice_features):
        """Test drift is calculated between versions."""
        profile_v1 = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        # Modify features slightly
        modified_features = sample_voice_features
        modified_features.stylistic_profile.formality_score = 85.0
        
        profile_v2 = builder.build(
            brand_id="brand123",
            voice_features=modified_features,
            platform="linkedin",
            previous_profile=profile_v1,
        )
        
        assert 0 <= profile_v2.drift_from_previous <= 1.0
    
    def test_signature_uniqueness(self, builder, sample_voice_features):
        """Test different profiles have different signatures."""
        profile1 = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        # Create different features
        different_features = sample_voice_features
        different_features.stylistic_profile.formality_score = 50.0
        
        profile2 = builder.build(
            brand_id="brand456",
            voice_features=different_features,
            platform="twitter",
        )
        
        assert profile1.signature != profile2.signature
    
    def test_platform_profiles_preserved(self, builder, sample_voice_features):
        """Test platform variations are preserved."""
        sample_voice_features.platform_variations = {
            "linkedin": {"formality": 80.0},
            "twitter": {"formality": 60.0},
        }
        
        profile = builder.build(
            brand_id="brand123",
            voice_features=sample_voice_features,
            platform="linkedin",
        )
        
        assert profile.platform_profiles == sample_voice_features.platform_variations