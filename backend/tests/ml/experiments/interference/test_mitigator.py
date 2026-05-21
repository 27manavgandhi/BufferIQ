"""Tests for interference mitigator."""

from bufferiq.ml.experiments.interference.mitigator import InterferenceMitigator


class TestInterferenceMitigator:
    """Test InterferenceMitigator."""

    def setup_method(self):
        """Setup test."""
        self.mitigator = InterferenceMitigator()

    def test_recommend_mitigation(self):
        """Test mitigation recommendation."""
        strategy = self.mitigator.recommend_mitigation(
            has_interference=True,
            num_clusters=20,
            mean_cluster_size=50,
        )

        assert "strategy" in strategy
        assert "reason" in strategy

    def test_create_buffer_zones(self):
        """Test buffer zone creation."""
        result = self.mitigator.create_buffer_zones(
            treatment_user_ids=["u1", "u2"],
            control_user_ids=["u3", "u4", "u5"],
            edges=[("u1", "u3"), ("u2", "u4")],
            buffer_distance=1,
        )

        assert "treatment" in result
        assert "control" in result
        assert "buffer" in result

    def test_assign_clusters(self):
        """Test cluster assignment."""
        clusters = [{"u1", "u2"}, {"u3", "u4"}, {"u5", "u6"}]

        assignments = self.mitigator.assign_clusters(clusters)

        assert "treatment" in assignments
        assert "control" in assignments
        # Should have users from all clusters
        total = len(assignments["treatment"]) + len(assignments["control"])
        assert total == 6
