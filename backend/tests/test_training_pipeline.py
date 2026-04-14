"""Tests for training pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from bufferiq.ml.training.config_schema import (
    TrainingPipelineConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
)
from bufferiq.ml.training.pipeline import TrainingPipeline


class TestTrainingPipeline:
    """Test training pipeline."""

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create test config."""
        return TrainingPipelineConfig(
            data=DataConfig(
                target_column="engagement_rate",
                platforms=["linkedin", "twitter"],
                test_size=0.2,
            ),
            model=ModelConfig(
                model_type="random_forest",
                hyperparameters={"n_estimators": 10},
            ),
            training=TrainingConfig(max_epochs=10, early_stopping_patience=3),
            experiment=ExperimentConfig(
                experiment_name="test_pipeline", description="Test"
            ),
        )

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    def test_init(
        self, config: TrainingPipelineConfig, mock_session: AsyncMock
    ) -> None:
        """Test initialization."""
        pipeline = TrainingPipeline(config, mock_session)

        assert pipeline.config == config
        assert pipeline.session == mock_session

    @pytest.mark.asyncio
    async def test_load_data(
        self, config: TrainingPipelineConfig, mock_session: AsyncMock
    ) -> None:
        """Test data loading."""
        # Mock database response
        mock_post = MagicMock()
        mock_post.id = 1
        mock_post.platform = "linkedin"
        mock_post.content = "test"
        mock_post.likes = 10
        mock_post.comments = 2
        mock_post.shares = 1
        mock_post.impressions = 100
        mock_post.clicks = 5

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_post]
        mock_session.execute.return_value = mock_result

        pipeline = TrainingPipeline(config, mock_session)
        df = await pipeline._load_data()

        assert len(df) > 0
        assert "engagement_rate" in df.columns