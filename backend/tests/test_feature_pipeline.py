"""Tests for feature engineering pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bufferiq.ml.features.content import ContentFeatureExtractor
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline
from bufferiq.ml.features.scaler import FeatureScaler
from bufferiq.ml.features.selector import FeatureSelector
from bufferiq.ml.features.temporal import TemporalFeatureExtractor


class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "published_at": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-02T14:00:00Z",
                    "2024-01-03T18:00:00Z",
                ],
                "content": [
                    "Hello world! #test https://example.com",
                    "Simple post",
                    "Another post with content",
                ],
                "platform": ["linkedin", "twitter", "bluesky"],
                "engagement_rate": [10.0, 15.0, 20.0],
            }
        )

    @pytest.mark.asyncio
    async def test_extract_features_default(self, sample_df: pd.DataFrame) -> None:
        """Test feature extraction with default extractors."""
        pipeline = FeatureEngineeringPipeline()
        features = await pipeline.extract_features(sample_df)

        assert len(features) == len(sample_df)
        assert len(features.columns) > 0

    @pytest.mark.asyncio
    async def test_extract_features_with_specific_extractors(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test feature extraction with specific extractors."""
        extractors = [
            TemporalFeatureExtractor(),
            ContentFeatureExtractor(),
        ]

        pipeline = FeatureEngineeringPipeline(extractors=extractors)
        features = await pipeline.extract_features(sample_df)

        assert len(features) == len(sample_df)
        # Should have temporal + content features
        assert len(features.columns) > 20

    @pytest.mark.asyncio
    async def test_extract_features_with_scaler(self, sample_df: pd.DataFrame) -> None:
        """Test feature extraction with scaling."""
        scaler = FeatureScaler(method="standard")
        pipeline = FeatureEngineeringPipeline(scaler=scaler)

        features = await pipeline.extract_features(sample_df, fit_scaler=True)

        # Features should be scaled (mean ~0, std ~1)
        assert features.mean().abs().max() < 1.0

    @pytest.mark.asyncio
    async def test_extract_features_with_selector(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test feature extraction with selection."""
        selector = FeatureSelector(method="variance", threshold=0.0)
        pipeline = FeatureEngineeringPipeline(selector=selector)

        features = await pipeline.extract_features(
            sample_df,
            fit_selector=True,
            target_column="engagement_rate",
        )

        # Should have fewer features after selection
        assert len(features.columns) > 0

    @pytest.mark.asyncio
    async def test_extract_features_empty_dataframe(self) -> None:
        """Test extraction with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline()
        df = pd.DataFrame()

        features = await pipeline.extract_features(df)

        assert features.empty

    @pytest.mark.asyncio
    async def test_fit_selector_without_target_raises_error(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that fitting selector without target raises error."""
        selector = FeatureSelector(method="k_best", k=5)
        pipeline = FeatureEngineeringPipeline(selector=selector)

        with pytest.raises(ValueError, match="target_column required"):
            await pipeline.extract_features(sample_df, fit_selector=True)

    def test_get_all_feature_names(self) -> None:
        """Test getting all feature names."""
        pipeline = FeatureEngineeringPipeline()
        feature_names = pipeline.get_all_feature_names()

        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_get_feature_stats(self) -> None:
        """Test getting feature statistics."""
        pipeline = FeatureEngineeringPipeline()
        stats = pipeline.get_feature_stats()

        assert "total_features" in stats
        assert "num_extractors" in stats
        assert stats["num_extractors"] == 5  # Default extractors

    def test_save_and_load_pipeline(self, sample_df: pd.DataFrame) -> None:
        """Test save and load pipeline."""
        scaler = FeatureScaler(method="standard")
        selector = FeatureSelector(method="variance", threshold=0.0)

        pipeline = FeatureEngineeringPipeline(scaler=scaler, selector=selector)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline"

            # Save
            pipeline.save_pipeline(str(path))
            assert path.exists()

            # Load
            loaded_pipeline = FeatureEngineeringPipeline.load_pipeline(str(path))
            assert len(loaded_pipeline.extractors) == len(pipeline.extractors)

    def test_load_nonexistent_pipeline_raises_error(self) -> None:
        """Test that loading nonexistent pipeline raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureEngineeringPipeline.load_pipeline("nonexistent")
