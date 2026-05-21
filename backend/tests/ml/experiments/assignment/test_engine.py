"""Tests for assignment engine."""

import pytest
from unittest.mock import Mock
from bufferiq.ml.experiments.assignment.engine import AssignmentEngine
from bufferiq.ml.experiments.design.designer import ExperimentDesigner, Variant, MetricType


class TestAssignmentEngine:
    """Test AssignmentEngine."""

    def setup_method(self):
        """Setup test."""
        self.db = Mock()
        self.engine = AssignmentEngine(self.db)

        designer = ExperimentDesigner()
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

        self.config = designer.design(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

    def test_assign_user(self):
        """Test user assignment."""
        assignment = self.engine.assign(
            experiment_config=self.config,
            user_id="user123",
        )

        assert assignment.experiment_id == self.config.experiment_id
        assert assignment.user_id == "user123"
        assert assignment.variant_id in ["control", "treatment"]
        assert assignment.is_new_assignment is True

    def test_assign_consistency(self):
        """Test assignment consistency."""
        assignment1 = self.engine.assign(
            experiment_config=self.config,
            user_id="user123",
        )

        assignment2 = self.engine.assign(
            experiment_config=self.config,
            user_id="user123",
        )

        assert assignment1.variant_id == assignment2.variant_id
        assert assignment2.is_new_assignment is False

    def test_force_variant(self):
        """Test forced variant assignment."""
        assignment = self.engine.assign(
            experiment_config=self.config,
            user_id="user123",
            force_variant="treatment",
        )

        assert assignment.variant_id == "treatment"

    def test_platform_validation(self):
        """Test platform validation."""
        with pytest.raises(ValueError, match="not supported"):
            self.engine.assign(
                experiment_config=self.config,
                user_id="user123",
                platform="facebook",
            )