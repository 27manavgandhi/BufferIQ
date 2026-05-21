"""Tests for assignment logger."""

from unittest.mock import Mock
from datetime import datetime
from bufferiq.ml.experiments.assignment.logger import AssignmentLogger
from bufferiq.ml.experiments.assignment.engine import Assignment


class TestAssignmentLogger:
    """Test AssignmentLogger."""

    def setup_method(self):
        """Setup test."""
        self.db = Mock()
        self.logger = AssignmentLogger(self.db)

    def test_log_assignment(self):
        """Test logging assignment."""
        assignment = Assignment(
            experiment_id="exp_001",
            user_id="user123",
            variant_id="treatment",
            variant_name="Treatment",
            assigned_at=datetime.now(),
            assignment_hash="abc123",
            is_new_assignment=True,
        )

        self.logger.log(assignment)

        # Should be in cache
        retrieved = self.logger.get_assignment("exp_001", "user123")
        assert retrieved is not None
        assert retrieved.user_id == "user123"

    def test_get_nonexistent_assignment(self):
        """Test getting non-existent assignment."""
        result = self.logger.get_assignment("exp_001", "user999")
        assert result is None

    def test_list_assignments(self):
        """Test listing assignments."""
        assignments = [
            Assignment(
                experiment_id="exp_001",
                user_id=f"user{i}",
                variant_id="treatment",
                variant_name="Treatment",
                assigned_at=datetime.now(),
                assignment_hash=f"hash{i}",
                is_new_assignment=True,
            )
            for i in range(5)
        ]

        for assignment in assignments:
            self.logger.log(assignment)

        listed = self.logger.list_assignments("exp_001")
        assert len(listed) == 5
