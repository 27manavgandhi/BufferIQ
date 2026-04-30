"""Tests for feature service."""

import pytest
from datetime import datetime

from bufferiq.api.services.feature_service import FeatureService
from bufferiq.api.models.prediction import PredictionRequest


@pytest.fixture
def feature_service():
    """Create feature service."""
    return FeatureService()


@pytest.fixture
def sample_request():
    """Sample prediction request."""
    return PredictionRequest(
        content="Just shipped a new feature! #tech @company",
        platform="linkedin",
        scheduled_time=datetime(2026, 4, 30, 14, 0, 0),
        has_media=True,
        has_link=True,
    )


@pytest.mark.asyncio
async def test_extract_features(feature_service, sample_request):
    """Test feature extraction."""
    features = await feature_service.extract_features(sample_request)

    assert len(features) == 92
    assert all(isinstance(f, float) for f in features)


@pytest.mark.asyncio
async def test_extract_features_caching(feature_service, sample_request):
    """Test feature caching."""
    # First extraction
    features1 = await feature_service.extract_features(sample_request)

    # Second extraction should use cache
    features2 = await feature_service.extract_features(sample_request)

    assert features1 == features2


@pytest.mark.asyncio
async def test_extract_features_content_length(feature_service):
    """Test content length feature."""
    request = PredictionRequest(
        content="Short",
        platform="linkedin",
    )

    features = await feature_service.extract_features(request)
    assert features[0] == 5  # Length of "Short"


@pytest.mark.asyncio
async def test_extract_features_hashtags(feature_service):
    """Test hashtag counting."""
    request = PredictionRequest(
        content="Post with #tech #ai #ml hashtags",
        platform="linkedin",
    )

    features = await feature_service.extract_features(request)
    assert features[5] == 3  # Three hashtags


@pytest.mark.asyncio
async def test_extract_features_mentions(feature_service):
    """Test mention counting."""
    request = PredictionRequest(
        content="Shoutout to @user1 and @user2",
        platform="linkedin",
    )

    features = await feature_service.extract_features(request)
    assert features[6] == 2  # Two mentions


@pytest.mark.asyncio
async def test_extract_features_platform_encoding(feature_service):
    """Test platform one-hot encoding."""
    for platform in ["linkedin", "twitter", "bluesky"]:
        request = PredictionRequest(
            content="Test",
            platform=platform,
        )

        features = await feature_service.extract_features(request)

        # Features 7, 8, 9 are platform one-hot encoding
        platform_features = features[7:10]
        assert sum(platform_features) == 1.0  # Only one should be 1


@pytest.mark.asyncio
async def test_extract_features_media_flag(feature_service):
    """Test media flag extraction."""
    request = PredictionRequest(
        content="Test",
        platform="linkedin",
        has_media=True,
    )

    features = await feature_service.extract_features(request)
    assert features[10] == 1.0  # Media flag


@pytest.mark.asyncio
async def test_extract_features_link_flag(feature_service):
    """Test link flag extraction."""
    request = PredictionRequest(
        content="Test",
        platform="linkedin",
        has_link=True,
    )

    features = await feature_service.extract_features(request)
    assert features[11] == 1.0  # Link flag


@pytest.mark.asyncio
async def test_clear_cache(feature_service, sample_request):
    """Test cache clearing."""
    # Extract features
    await feature_service.extract_features(sample_request)
    assert len(feature_service._cache) > 0

    # Clear cache
    feature_service.clear_cache()
    assert len(feature_service._cache) == 0


@pytest.mark.asyncio
async def test_batch_extraction(feature_service):
    """Test batch feature extraction."""
    requests = [
        PredictionRequest(content=f"Post {i}", platform="linkedin")
        for i in range(5)
    ]

    features_list = await feature_service.extract_batch_features(requests)

    assert len(features_list) == 5
    assert all(len(f) == 92 for f in features_list)