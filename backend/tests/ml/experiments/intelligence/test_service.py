"""Tests for experiment intelligence service."""

import pytest
import numpy as np
from unittest.mock import Mock
from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService
from bufferiq.ml.experiments.design.designer import Variant, MetricType


class TestExperimentIntelligenceService:
    """Test ExperimentIntelligenceService."""

    def setup_method(self):
        """Setup test."""
        self.db = Mock()
        self.service = ExperimentIntelligenceService(self.db)

    @pytest.mark.asyncio
    async def test_create_experiment(self):
        """Test experiment creation."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        config = await self.service.create_experiment(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        assert config.experiment_id.startswith("exp_")
        assert config.name == "Test"

    @pytest.mark.asyncio
    async def test_assign_user(self):
        """Test user assignment."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        config = await self.service.create_experiment(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        assignment = await self.service.assign_user(
            experiment_id=config.experiment_id,
            user_id="user123",
        )

        assert assignment.user_id == "user123"
        assert assignment.variant_id in ["control", "treatment"]

    @pytest.mark.asyncio
    async def test_track_metric(self):
        """Test metric tracking."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        config = await self.service.create_experiment(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        assignment = await self.service.assign_user(
            experiment_id=config.experiment_id,
            user_id="user123",
        )

        await self.service.track_metric(
            experiment_id=config.experiment_id,
            user_id="user123",
            metric_type=MetricType.ENGAGEMENT_RATE,
            value=1.0,
            variant_id=assignment.variant_id,
        )

        # Should not raise error

    @pytest.mark.asyncio
    async def test_analyze_experiment(self):
        """Test experiment analysis."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        config = await self.service.create_experiment(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        # Track metrics for many users
        np.random.seed(42)
        for i in range(200):
            user_id = f"user{i}"
            assignment = await self.service.assign_user(
                experiment_id=config.experiment_id,
                user_id=user_id,
            )

            value = (
                np.random.binomial(1, 0.05)
                if assignment.variant_id == "control"
                else np.random.binomial(1, 0.06)
            )

            await self.service.track_metric(
                experiment_id=config.experiment_id,
                user_id=user_id,
                metric_type=MetricType.ENGAGEMENT_RATE,
                value=float(value),
                variant_id=assignment.variant_id,
            )

        # Analyze
        results = await self.service.analyze_experiment(
            experiment_id=config.experiment_id,
            min_sample_size=50,
        )

        assert results["status"] == "complete"
        assert "statistical_result" in results

    @pytest.mark.asyncio
    async def test_platform_validation(self):
        """Test platform validation."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        with pytest.raises(ValueError, match="not supported"):
            await self.service.create_experiment(
                name="Invalid",
                description="Invalid",
                variants=variants,
                platform="facebook",
                primary_metric=MetricType.ENGAGEMENT_RATE,
                baseline_rate=0.05,
            )
