"""Tests for experiment service."""

import pytest
from unittest.mock import Mock
from bufferiq.api.services.experiment_service import ExperimentService


class TestExperimentService:
    """Test ExperimentService."""

    def setup_method(self):
        """Setup test."""
        self.db = Mock()
        self.service = ExperimentService(self.db)

    # Service tests would mirror intelligence service tests
    pass
