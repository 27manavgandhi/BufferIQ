"""Tests for prediction service."""

import pytest
from unittest.mock import Mock, AsyncMock
import numpy as np

from bufferiq.api.services.prediction_service import PredictionService
from bufferiq.api.models.prediction import PredictionRequest


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([7.5]))
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    """Create mock model loader."""
    loader = Mock()
    loader.load_model = Mock(return_value=mock_model)
    return loader


@pytest.fixture
def mock_feature_service():
    """Create mock feature service."""
    service = Mock()
    service.extract_features = AsyncMock(return_value=[1.0] * 92)
    return service


@pytest.fixture
def prediction_service(mock_model_loader, mock_feature_service):
    """Create prediction service."""
    return PredictionService(
        model_loader=mock_model_loader,
        feature_service=mock_feature_service,
    )


@pytest.fixture
def sample_request():
    """Sample prediction request."""
    return PredictionRequest(
        content="Test post",
        platform="linkedin",
    )


@pytest.mark.asyncio
async def test_predict_success(prediction_service, sample_request):
    """Test successful prediction."""
    response = await prediction_service.predict(sample_request)

    assert response.engagement_score > 0
    assert 0 <= response.confidence <= 1
    assert response.breakdown is not None
    assert response.metadata is not None


@pytest.mark.asyncio
async def test_predict_calls_feature_service(
    prediction_service, sample_request, mock_feature_service
):
    """Test prediction calls feature service."""
    await prediction_service.predict(sample_request)

    mock_feature_service.extract_features.assert_called_once()


@pytest.mark.asyncio
async def test_predict_calls_model_loader(
    prediction_service, sample_request, mock_model_loader
):
    """Test prediction calls model loader."""
    await prediction_service.predict(sample_request)

    mock_model_loader.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_predict_with_different_models(
    prediction_service, sample_request
):
    """Test prediction with different model names."""
    for model_name in ["xgboost", "lightgbm", "ensemble"]:
        response = await prediction_service.predict(
            sample_request, model_name
        )
        assert response.metadata.model_version == model_name


@pytest.mark.asyncio
async def test_predict_breakdown_sums_to_score(
    prediction_service, sample_request
):
    """Test breakdown approximately sums to total score."""
    response = await prediction_service.predict(sample_request)

    breakdown_sum = (
        response.breakdown.likes
        + response.breakdown.comments
        + response.breakdown.shares
    )

    # Should be approximately equal
    assert abs(breakdown_sum - response.engagement_score) < 0.1


@pytest.mark.asyncio
async def test_predict_platform_specific_breakdown(
    prediction_service
):
    """Test platform-specific breakdown distributions."""
    platforms = ["linkedin", "twitter", "bluesky"]

    for platform in platforms:
        request = PredictionRequest(
            content="Test post",
            platform=platform,
        )

        response = await prediction_service.predict(request)

        # Breakdown should exist for all platforms
        assert response.breakdown.likes > 0
        assert response.breakdown.comments >= 0
        assert response.breakdown.shares >= 0


@pytest.mark.asyncio
async def test_predict_metadata_structure(
    prediction_service, sample_request
):
    """Test metadata structure."""
    response = await prediction_service.predict(sample_request)

    assert response.metadata.model_version is not None
    assert response.metadata.features_used == 92
    assert response.metadata.cached is False


@pytest.mark.asyncio
async def test_predict_error_handling(
    prediction_service, sample_request, mock_model_loader
):
    """Test error handling."""
    # Make model loader raise error
    mock_model_loader.load_model.side_effect = Exception("Model error")

    with pytest.raises(ValueError, match="Prediction failed"):
        await prediction_service.predict(sample_request)


@pytest.mark.asyncio
async def test_predict_confidence_range(
    prediction_service, sample_request
):
    """Test confidence is in valid range."""
    response = await prediction_service.predict(sample_request)

    assert 0 <= response.confidence <= 1


@pytest.mark.asyncio
async def test_predict_with_scheduled_time(prediction_service):
    """Test prediction with scheduled time."""
    from datetime import datetime

    request = PredictionRequest(
        content="Test post",
        platform="linkedin",
        scheduled_time=datetime(2026, 4, 30, 14, 0, 0),
    )

    response = await prediction_service.predict(request)
    assert response is not None


@pytest.mark.asyncio
async def test_predict_with_media(prediction_service):
    """Test prediction with media flag."""
    request = PredictionRequest(
        content="Test post",
        platform="linkedin",
        has_media=True,
    )

    response = await prediction_service.predict(request)
    assert response is not None


@pytest.mark.asyncio
async def test_predict_with_link(prediction_service):
    """Test prediction with link flag."""
    request = PredictionRequest(
        content="Test post",
        platform="linkedin",
        has_link=True,
    )

    response = await prediction_service.predict(request)
    assert response is not None