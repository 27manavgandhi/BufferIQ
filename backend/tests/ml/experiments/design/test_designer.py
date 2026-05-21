"""Tests for experiment designer."""

import pytest
import numpy as np

from bufferiq.ml.experiments.design.designer import (
    ExperimentDesigner,
    Variant,
    ExperimentType,
    MetricType,
)


class TestExperimentDesigner:
    """Test ExperimentDesigner."""

    def setup_method(self):
        """Setup test."""
        self.designer = ExperimentDesigner()

    def test_design_ab_test(self):
        """Test A/B test design."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {"version": "new"}),
        ]

        config = self.designer.design(
            name="Test",
            description="Test experiment",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
            mde=0.10,
        )

        assert config.experiment_id.startswith("exp_")
        assert config.name == "Test"
        assert config.platform == "linkedin"
        assert config.type == ExperimentType.AB_TEST
        assert len(config.variants) == 2
        assert config.required_sample_size > 0

    def test_design_multivariate(self):
        """Test multivariate design."""
        variants = [
            Variant("control", "Control", "Original", 0.33, {}, True),
            Variant("treatment_a", "Treatment A", "New A", 0.33, {"version": "a"}),
            Variant("treatment_b", "Treatment B", "New B", 0.34, {"version": "b"}),
        ]

        config = self.designer.design(
            name="MVT",
            description="Multivariate test",
            variants=variants,
            platform="twitter",
            primary_metric=MetricType.CLICK_THROUGH_RATE,
            baseline_rate=0.02,
        )

        assert config.type == ExperimentType.MULTIVARIATE
        assert len(config.variants) == 3

    def test_validate_variants_traffic_allocation(self):
        """Test traffic allocation validation."""
        variants = [
            Variant("control", "Control", "Original", 0.6, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        with pytest.raises(ValueError, match="sum to 1.0"):
            self.designer.design(
                name="Invalid",
                description="Invalid",
                variants=variants,
                platform="linkedin",
                primary_metric=MetricType.ENGAGEMENT_RATE,
                baseline_rate=0.05,
            )

    def test_validate_variants_control_count(self):
        """Test control count validation."""
        variants = [
            Variant("control_a", "Control A", "Original A", 0.5, {}, True),
            Variant("control_b", "Control B", "Original B", 0.5, {}, True),
        ]

        with pytest.raises(ValueError, match="one variant must be control"):
            self.designer.design(
                name="Invalid",
                description="Invalid",
                variants=variants,
                platform="linkedin",
                primary_metric=MetricType.ENGAGEMENT_RATE,
                baseline_rate=0.05,
            )

    def test_platform_validation(self):
        """Test platform validation."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        with pytest.raises(ValueError, match="not supported"):
            self.designer.design(
                name="Invalid Platform",
                description="Invalid",
                variants=variants,
                platform="facebook",
                primary_metric=MetricType.ENGAGEMENT_RATE,
                baseline_rate=0.05,
            )

    def test_create_ab_test_helper(self):
        """Test create_ab_test helper."""
        config = self.designer.create_ab_test(
            name="Quick AB",
            description="Quick test",
            control_name="Control",
            treatment_name="Treatment",
            treatment_changes={"headline": "new"},
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        assert len(config.variants) == 2
        assert config.type == ExperimentType.AB_TEST

    def test_estimate_duration(self):
        """Test duration estimation."""
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        config = self.designer.design(
            name="Duration Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
            expected_daily_traffic=10000,
        )

        assert config.estimated_duration_days >= 7