"""Tests for topic extractor."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from bufferiq.ml.gaps.topics.extractor import TopicExtractor, Topic, TopicCluster


class TestTopicExtractor:
    """Test TopicExtractor class."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def extractor(self, mock_db):
        """Create extractor instance."""
        return TopicExtractor(mock_db, min_topic_posts=3)

    @pytest.mark.asyncio
    async def test_extract_topics_success(self, extractor):
        """Test successful topic extraction."""
        topics = await extractor.extract(
            user_id="user123",
            platform="linkedin",
            lookback_days=90
        )

        assert len(topics) > 0
        assert all(isinstance(t, Topic) for t in topics)
        assert all(t.post_count >= extractor.min_posts for t in topics)

    @pytest.mark.asyncio
    async def test_extract_invalid_platform(self, extractor):
        """Test extraction with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await extractor.extract(
                user_id="user123",
                platform="facebook",
                lookback_days=90
            )

    @pytest.mark.asyncio
    async def test_extract_insufficient_posts(self, extractor, mock_db):
        """Test extraction with insufficient posts."""
        # Mock to return too few posts
        with patch.object(extractor, '_fetch_posts', return_value=[]):
            with pytest.raises(ValueError, match="Insufficient posts"):
                await extractor.extract(
                    user_id="user123",
                    platform="linkedin",
                    lookback_days=90
                )

    def test_calculate_growth_rate(self, extractor):
        """Test growth rate calculation."""
        now = datetime.now()
        posts = [
            {"created_at": now - timedelta(days=i), "engagement": 100}
            for i in range(60)
        ]

        growth = extractor._calculate_growth_rate(posts)

        assert isinstance(growth, float)
        assert -100 <= growth <= 1000  # Reasonable bounds

    def test_calculate_relevance(self, extractor):
        """Test relevance score calculation."""
        score = extractor._calculate_relevance(
            post_count=15,
            avg_engagement=250.0,
            growth_rate=25.0
        )

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_generate_topic_id(self, extractor):
        """Test topic ID generation."""
        keywords = ["machine", "learning", "AI"]
        
        topic_id = extractor._generate_topic_id(keywords)

        assert topic_id.startswith("topic_")
        assert len(topic_id) == 18  # topic_ + 12 char hash

        # Same keywords should generate same ID
        topic_id2 = extractor._generate_topic_id(keywords)
        assert topic_id == topic_id2

    def test_generate_topic_name(self, extractor):
        """Test topic name generation."""
        keywords = ["machine", "learning", "AI"]
        
        name = extractor._generate_topic_name(keywords)

        assert isinstance(name, str)
        assert len(name) > 0
        assert "Machine" in name or "Learning" in name

    def test_cluster_topics(self, extractor):
        """Test topic clustering."""
        import numpy as np

        # Mock topic vectors
        vectors = np.random.rand(5, 100)
        
        # Mock topic data
        topic_data = [
            {"topic_id": f"topic_{i}", "keywords": [f"kw{i}"], "post_count": 10}
            for i in range(5)
        ]

        clusters = extractor.cluster_topics(vectors, topic_data)

        assert isinstance(clusters, list)
        assert all(isinstance(c, TopicCluster) for c in clusters)

    @pytest.mark.asyncio
    async def test_platform_validation_linkedin(self, extractor):
        """Test platform validation for LinkedIn."""
        topics = await extractor.extract(
            user_id="user123",
            platform="linkedin",
            lookback_days=30
        )
        assert len(topics) >= 0

    @pytest.mark.asyncio
    async def test_platform_validation_twitter(self, extractor):
        """Test platform validation for Twitter."""
        topics = await extractor.extract(
            user_id="user123",
            platform="twitter",
            lookback_days=30
        )
        assert len(topics) >= 0

    @pytest.mark.asyncio
    async def test_platform_validation_bluesky(self, extractor):
        """Test platform validation for Bluesky."""
        topics = await extractor.extract(
            user_id="user123",
            platform="bluesky",
            lookback_days=30
        )
        assert len(topics) >= 0

    def test_topic_serialization(self, extractor):
        """Test topic to_dict method."""
        topic = Topic(
            topic_id="test123",
            name="Test Topic",
            keywords=["test", "topic"],
            description="Test description",
            cluster_id=1,
            lifecycle_stage="growing",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            post_count=10,
            total_engagement=500,
            avg_engagement=50.0,
            growth_rate=15.5,
            relevance_score=0.85
        )

        topic_dict = topic.to_dict()

        assert isinstance(topic_dict, dict)
        assert topic_dict["topic_id"] == "test123"
        assert topic_dict["name"] == "Test Topic"
        assert topic_dict["relevance_score"] == 0.85