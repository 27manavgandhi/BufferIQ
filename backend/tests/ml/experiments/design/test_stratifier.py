"""Tests for stratifier."""

from bufferiq.ml.experiments.design.stratifier import Stratifier


class TestStratifier:
    """Test Stratifier."""

    def setup_method(self):
        """Setup test."""
        self.stratifier = Stratifier()

    def test_create_strata(self):
        """Test strata creation."""
        users = [
            {"user_id": "u1", "segment": "new"},
            {"user_id": "u2", "segment": "new"},
            {"user_id": "u3", "segment": "returning"},
            {"user_id": "u4", "segment": "returning"},
        ]

        strata = self.stratifier.create_strata(users, "segment")

        assert len(strata) == 2
        assert "new" in strata
        assert "returning" in strata
        assert len(strata["new"]) == 2
        assert len(strata["returning"]) == 2

    def test_validate_balance(self):
        """Test balance validation."""
        users = [
            {"user_id": f"u{i}", "segment": "new" if i < 50 else "old"}
            for i in range(100)
        ]

        assignments = {
            "control": [f"u{i}" for i in range(0, 50, 2)] + [f"u{i}" for i in range(50, 100, 2)],
            "treatment": [f"u{i}" for i in range(1, 50, 2)] + [f"u{i}" for i in range(51, 100, 2)],
        }

        is_balanced = self.stratifier.validate_balance(
            assignments=assignments,
            stratification_key="segment",
            users=users,
        )

        assert is_balanced