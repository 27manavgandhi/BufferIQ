"""Tests for cache service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from bufferiq.api.services.cache_service import CacheService
from bufferiq.api.models.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionMetadata,
    EngagementScores,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.keys = AsyncMock(return_value=[])
    redis.delete = AsyncMock(return_value=0)
    redis.info = AsyncMock(return_value={})
    redis.dbsize = AsyncMock(return_value=0)
    redis.close = AsyncMock()
    return redis


@pytest.fixture
async def cache_service(mock_redis):
    """Create cache service with mock Redis."""
    with patch("aioredis.from_url", return_value=mock_redis):
        service = CacheService(redis_url="redis://localhost", ttl=3600)
        await service._ensure_connected()
        return service


@pytest.fixture
def sample_request():
    """Sample prediction request."""
    return PredictionRequest(
        content="Test post",
        platform="linkedin",
    )


@pytest.fixture
def sample_response():
    """Sample prediction response."""
    return PredictionResponse(
        engagement_score=7.5,
        confidence=0.85,
        breakdown=EngagementScores(likes=45, comments=8, shares=3),
        metadata=PredictionMetadata(
            model_version="test",
            inference_time_ms=50.0,
            features_used=92,
        ),
    )


@pytest.mark.asyncio
async def test_generate_key(cache_service, sample_request):
    """Test cache key generation."""
    key1 = cache_service.generate_key(sample_request)
    key2 = cache_service.generate_key(sample_request)

    assert key1 == key2  # Same request = same key
    assert key1.startswith("pred:")


@pytest.mark.asyncio
async def test_generate_key_deterministic(cache_service):
    """Test key generation is deterministic."""
    request1 = PredictionRequest(content="Test", platform="linkedin")
    request2 = PredictionRequest(content="Test", platform="linkedin")

    key1 = cache_service.generate_key(request1)
    key2 = cache_service.generate_key(request2)

    assert key1 == key2


@pytest.mark.asyncio
async def test_generate_key_different_content(cache_service):
    """Test different content generates different keys."""
    request1 = PredictionRequest(content="Test 1", platform="linkedin")
    request2 = PredictionRequest(content="Test 2", platform="linkedin")

    key1 = cache_service.generate_key(request1)
    key2 = cache_service.generate_key(request2)

    assert key1 != key2


@pytest.mark.asyncio
async def test_generate_key_different_platform(cache_service):
    """Test different platform generates different keys."""
    request1 = PredictionRequest(content="Test", platform="linkedin")
    request2 = PredictionRequest(content="Test", platform="twitter")

    key1 = cache_service.generate_key(request1)
    key2 = cache_service.generate_key(request2)

    assert key1 != key2


@pytest.mark.asyncio
async def test_get_cache_miss(cache_service, mock_redis):
    """Test cache miss."""
    mock_redis.get.return_value = None

    result = await cache_service.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_get_cache_hit(cache_service, mock_redis, sample_response):
    """Test cache hit."""
    mock_redis.get.return_value = sample_response.json()

    result = await cache_service.get("test_key")
    assert result is not None
    assert result.engagement_score == sample_response.engagement_score


@pytest.mark.asyncio
async def test_set_cache(cache_service, mock_redis, sample_response):
    """Test setting cache."""
    await cache_service.set("test_key", sample_response)

    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_invalidate_cache(cache_service, mock_redis):
    """Test cache invalidation."""
    mock_redis.keys.return_value = ["pred:key1", "pred:key2"]
    mock_redis.delete.return_value = 2

    deleted = await cache_service.invalidate("pred:*")
    assert deleted == 2


@pytest.mark.asyncio
async def test_stats(cache_service, mock_redis):
    """Test cache statistics."""
    mock_redis.info.return_value = {
        "used_memory": 1024 * 1024,  # 1 MB
        "keyspace_hits": 100,
        "keyspace_misses": 20,
    }
    mock_redis.dbsize.return_value = 50

    stats = await cache_service.stats()

    assert stats["status"] == "connected"
    assert stats["keys"] == 50
    assert stats["hits"] == 100
    assert stats["misses"] == 20


@pytest.mark.asyncio
async def test_close(cache_service, mock_redis):
    """Test closing connection."""
    await cache_service.close()
    mock_redis.close.assert_called_once()