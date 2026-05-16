"""Tests for voice profile versioning."""

import pytest
from datetime import datetime
from bufferiq.ml.voice.profiler.versioning import VoiceProfileVersioning
from bufferiq.ml.voice.profiler.builder import VoiceProfile


class TestVoiceProfileVersioning:
    """Test voice profile versioning."""
    
    @pytest.fixture
    def versioning(self):
        """Create versioning instance."""
        return VoiceProfileVersioning()
    
    @pytest.fixture
    def sample_profile(self):
        """Create sample profile."""
        return VoiceProfile(
            profile_id="brand123_linkedin_v1_20240508",
            brand_id="brand123",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={"ttr": 0.8},
            syntactic_fingerprint={"avg_len": 15},
            stylistic_fingerprint={"formality": 80},
            signature="abc123",
            confidence=0.85,
            sample_size=50,
            platform_profiles={},
        )
    
    def test_add_version(self, versioning, sample_profile):
        """Test adding version to history."""
        versioning.add_version(sample_profile)
        
        assert "brand123" in versioning.version_history
        assert len(versioning.version_history["brand123"]) == 1
    
    def test_add_multiple_versions(self, versioning, sample_profile):
        """Test adding multiple versions."""
        versioning.add_version(sample_profile)
        
        profile_v2 = VoiceProfile(
            profile_id="brand123_linkedin_v2_20240509",
            brand_id="brand123",
            version=2,
            created_at=datetime.utcnow(),
            lexical_fingerprint={"ttr": 0.85},
            syntactic_fingerprint={"avg_len": 16},
            stylistic_fingerprint={"formality": 82},
            signature="def456",
            confidence=0.87,
            sample_size=60,
            platform_profiles={},
            previous_version="brand123_linkedin_v1_20240508",
        )
        
        versioning.add_version(profile_v2)
        
        assert len(versioning.version_history["brand123"]) == 2
    
    def test_get_latest_version(self, versioning, sample_profile):
        """Test getting latest version."""
        versioning.add_version(sample_profile)
        
        latest = versioning.get_latest_version("brand123")
        
        assert latest is not None
        assert latest.version == 1
    
    def test_get_latest_version_returns_highest(self, versioning, sample_profile):
        """Test latest version returns highest version number."""
        versioning.add_version(sample_profile)
        
        profile_v2 = VoiceProfile(
            profile_id="brand123_linkedin_v2_20240509",
            brand_id="brand123",
            version=2,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="def456",
            confidence=0.87,
            sample_size=60,
            platform_profiles={},
        )
        
        versioning.add_version(profile_v2)
        
        latest = versioning.get_latest_version("brand123")
        
        assert latest.version == 2
    
    def test_get_latest_version_no_history_returns_none(self, versioning):
        """Test getting latest version with no history."""
        latest = versioning.get_latest_version("nonexistent")
        
        assert latest is None
    
    def test_get_version_history(self, versioning, sample_profile):
        """Test getting complete version history."""
        versioning.add_version(sample_profile)
        
        profile_v2 = VoiceProfile(
            profile_id="brand123_linkedin_v2_20240509",
            brand_id="brand123",
            version=2,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="def456",
            confidence=0.87,
            sample_size=60,
            platform_profiles={},
        )
        
        versioning.add_version(profile_v2)
        
        history = versioning.get_version_history("brand123")
        
        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2
    
    def test_get_version_history_empty(self, versioning):
        """Test getting history for non-existent brand."""
        history = versioning.get_version_history("nonexistent")
        
        assert history == []
    
    def test_calculate_total_drift(self, versioning):
        """Test calculating total drift."""
        profile_v1 = VoiceProfile(
            profile_id="brand123_linkedin_v1_20240508",
            brand_id="brand123",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="abc",
            confidence=0.85,
            sample_size=50,
            platform_profiles={},
        )
        
        profile_v2 = VoiceProfile(
            profile_id="brand123_linkedin_v2_20240509",
            brand_id="brand123",
            version=2,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="def",
            confidence=0.87,
            sample_size=60,
            platform_profiles={},
            drift_from_previous=0.15,
        )
        
        profile_v3 = VoiceProfile(
            profile_id="brand123_linkedin_v3_20240510",
            brand_id="brand123",
            version=3,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="ghi",
            confidence=0.88,
            sample_size=70,
            platform_profiles={},
            drift_from_previous=0.10,
        )
        
        versioning.add_version(profile_v1)
        versioning.add_version(profile_v2)
        versioning.add_version(profile_v3)
        
        total_drift = versioning.calculate_total_drift("brand123")
        
        assert total_drift == pytest.approx(0.25)  # 0.15 + 0.10
    
    def test_calculate_total_drift_no_history(self, versioning):
        """Test calculating drift with no history."""
        drift = versioning.calculate_total_drift("nonexistent")
        
        assert drift == 0.0
    
    def test_should_create_new_version_high_drift(self, versioning):
        """Test should create new version with high drift."""
        profile = VoiceProfile(
            profile_id="brand123_linkedin_v1_20240508",
            brand_id="brand123",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="abc",
            confidence=0.85,
            sample_size=50,
            platform_profiles={},
            drift_from_previous=0.20,  # High drift
        )
        
        versioning.add_version(profile)
        
        should_create = versioning.should_create_new_version("brand123")
        
        assert should_create is True
    
    def test_should_create_new_version_low_drift(self, versioning):
        """Test should not create new version with low drift."""
        profile = VoiceProfile(
            profile_id="brand123_linkedin_v1_20240508",
            brand_id="brand123",
            version=1,
            created_at=datetime.utcnow(),
            lexical_fingerprint={},
            syntactic_fingerprint={},
            stylistic_fingerprint={},
            signature="abc",
            confidence=0.85,
            sample_size=50,
            platform_profiles={},
            drift_from_previous=0.05,  # Low drift
        )
        
        versioning.add_version(profile)
        
        should_create = versioning.should_create_new_version("brand123")
        
        assert should_create is False
    
    def test_should_create_new_version_no_history(self, versioning):
        """Test should create version with no history."""
        should_create = versioning.should_create_new_version("new_brand")
        
        assert should_create is True