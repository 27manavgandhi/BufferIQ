"""Tests for voice extractor."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from bufferiq.ml.voice.extraction.extractor import VoiceExtractor, VoiceFeatures


class TestVoiceExtractor:
    """Test voice extractor."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def extractor(self, mock_db):
        """Create extractor instance."""
        return VoiceExtractor(mock_db)
    
    @pytest.mark.asyncio
    async def test_extract_basic(self, extractor):
        """Test basic voice extraction."""
        features = await extractor.extract(
            user_id="user123",
            platform="linkedin",
            lookback_days=30,
            min_posts=10,
        )
        
        assert isinstance(features, VoiceFeatures)
        assert features.sample_size > 0
        assert 0 <= features.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_extract_invalid_platform_raises_error(self, extractor):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            await extractor.extract(
                user_id="user123",
                platform="facebook",
                lookback_days=30,
            )
    
    @pytest.mark.asyncio
    async def test_extract_insufficient_posts_raises_error(self, extractor):
        """Test insufficient posts raises error."""
        with patch.object(extractor, '_fetch_historical_posts', return_value=[]):
            with pytest.raises(ValueError, match="Insufficient posts"):
                await extractor.extract(
                    user_id="user123",
                    platform="linkedin",
                    min_posts=20,
                )
    
    @pytest.mark.asyncio
    async def test_extract_creates_all_profiles(self, extractor):
        """Test extraction creates all profile types."""
        features = await extractor.extract(
            user_id="user123",
            platform="linkedin",
        )
        
        assert features.lexical_profile is not None
        assert features.syntactic_profile is not None
        assert features.stylistic_profile is not None
    
    @pytest.mark.asyncio
    async def test_extract_temporal_evolution(self, extractor):
        """Test temporal evolution analysis."""
        features = await extractor.extract(
            user_id="user123",
            platform="linkedin",
        )
        
        assert 'temporal_evolution' in features.temporal_evolution or features.temporal_evolution is not None
    
    @pytest.mark.asyncio
    async def test_extract_platform_variations(self, extractor):
        """Test platform variation analysis."""
        features = await extractor.extract(
            user_id="user123",
            platform="linkedin",
        )
        
        assert isinstance(features.platform_variations, dict)
    
    @pytest.mark.asyncio
    async def test_confidence_high_sample_size(self, extractor):
        """Test confidence is high for large sample."""
        mock_posts = [
            {"text": f"Post {i}", "created_at": datetime.utcnow(), "platform": "linkedin"}
            for i in range(100)
        ]
        
        with patch.object(extractor, '_fetch_historical_posts', return_value=mock_posts):
            features = await extractor.extract(
                user_id="user123",
                platform="linkedin",
            )
            
            assert features.confidence_score >= 0.9
    
    @pytest.mark.asyncio
    async def test_confidence_low_sample_size(self, extractor):
        """Test confidence is lower for small sample."""
        mock_posts = [
            {"text": f"Post {i}", "created_at": datetime.utcnow(), "platform": "linkedin"}
            for i in range(25)
        ]
        
        with patch.object(extractor, '_fetch_historical_posts', return_value=mock_posts):
            features = await extractor.extract(
                user_id="user123",
                platform="linkedin",
            )
            
            assert features.confidence_score < 0.9
    
    @pytest.mark.asyncio
    async def test_extraction_date_set(self, extractor):
        """Test extraction date is set."""
        features = await extractor.extract(
            user_id="user123",
            platform="linkedin",
        )
        
        assert isinstance(features.extraction_date, datetime)
    
    @pytest.mark.asyncio
    async def test_extract_twitter_platform(self, extractor):
        """Test extraction for Twitter platform."""
        features = await extractor.extract(
            user_id="user123",
            platform="twitter",
        )
        
        assert features.sample_size > 0
    
    @pytest.mark.asyncio
    async def test_extract_bluesky_platform(self, extractor):
        """Test extraction for Bluesky platform."""
        features = await extractor.extract(
            user_id="user123",
            platform="bluesky",
        )
        
        assert features.sample_size > 0
    
    def test_fetch_historical_posts_returns_list(self, extractor):
        """Test fetch returns list."""
        posts = extractor._fetch_historical_posts("user123", "linkedin", 30)
        
        assert isinstance(posts, list)
    
    def test_analyze_temporal_evolution_returns_dict(self, extractor):
        """Test temporal evolution returns dict."""
        posts = [
            {"text": "Post", "created_at": datetime.utcnow() - timedelta(days=i), "platform": "linkedin"}
            for i in range(60)
        ]
        
        evolution = extractor._analyze_temporal_evolution(posts)
        
        assert isinstance(evolution, dict)
        assert 'early_formality' in evolution
        assert 'recent_formality' in evolution
    
    def test_analyze_platform_variations_returns_dict(self, extractor):
        """Test platform variations returns dict."""
        posts = [
            {"text": "Post on LinkedIn", "platform": "linkedin"},
            {"text": "Post on Twitter", "platform": "twitter"},
        ] * 5
        
        variations = extractor._analyze_platform_variations(posts)
        
        assert isinstance(variations, dict)
    
    def test_calculate_confidence_min_posts(self, extractor):
        """Test confidence at minimum posts."""
        confidence = extractor._calculate_confidence(20, 20)
        assert confidence == 0.6
    
    def test_calculate_confidence_below_min(self, extractor):
        """Test confidence below minimum."""
        confidence = extractor._calculate_confidence(10, 20)
        assert confidence == 0.0
    
    def test_calculate_confidence_high_sample(self, extractor):
        """Test confidence with high sample."""
        confidence = extractor._calculate_confidence(100, 20)
        assert confidence == 0.95
    
    def test_calculate_confidence_medium_sample(self, extractor):
        """Test confidence with medium sample."""
        confidence = extractor._calculate_confidence(50, 20)
        assert confidence == 0.85